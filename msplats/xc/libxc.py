from functools import partial

import equinox as eqx
import jax
from einops import rearrange
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float
from pyscf.dft import libxc

from .base import XCEnergyDensity

AOMatrix = Float[Array, "s nao nao"] | Float[Array, "nao nao"]
ScalarFeature = Float[Array, "... s"] | Float[Array, "..."]
VectorFeature = Float[Array, "... i s"] | Float[Array, "... i"]


def _eval_xc_libxc(xc, rho_data, deriv, spin):

    if spin:
        *batch_shape, n_rho_features, _ = rho_data.shape
        out_shape = (n_rho_features, 2) * deriv
        rho_data = rearrange(rho_data, "... r s -> (...) r s")
    else:
        *batch_shape, n_rho_features = rho_data.shape
        out_shape = (n_rho_features,) * deriv
        rho_data = rearrange(rho_data, "... r -> (...) r")

    exc_data = libxc.eval_xc_eff(xc, rho_data.transpose(), deriv).transpose()

    if len(batch_shape) > 1:
        exc_data = exc_data.reshape(*batch_shape, *out_shape)
    elif not batch_shape:
        exc_data = exc_data.squeeze(axis=0)

    return exc_data


def _eval_xc(xc, rho_data, deriv=0, spin=True):

    if spin:
        *batch_shape, n_grad_features, _ = rho_data.shape
        out_shape = tuple(batch_shape) + (n_grad_features, 2) * deriv
    else:
        *batch_shape, n_grad_features = rho_data.shape
        out_shape = tuple(batch_shape) + (n_grad_features,) * deriv

    fn = lambda rho_data: _eval_xc_libxc(xc, rho_data, deriv, spin)
    out_shape_dtype = jax.ShapeDtypeStruct(out_shape, rho_data.dtype)

    return jax.pure_callback(fn, out_shape_dtype, rho_data, vmap_method="broadcast_all")


def _stack_density_data(rho, grad_rho=None, tau=None, spin=True):

    rho_data = []

    if spin:
        rho_data.append(rearrange(rho, "... s -> ... 1 s"))

        if grad_rho is not None:
            rho_data.append(grad_rho)
        if tau is not None:
            rho_data.append(rearrange(tau, "... s -> ... 1 s"))

        return jnp.concatenate(rho_data, axis=-2)

    else:
        rho_data.append(rho[..., None])

        if grad_rho is not None:
            rho_data.append(grad_rho)
        if tau is not None:
            rho_data.append(tau[..., None])

        return jnp.concatenate(rho_data, axis=-1)


def _total_rho(rho_data, spin):
    return rho_data[..., 0, :].sum(axis=-1) if spin else rho_data[..., 0]


@partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def _xc_energy_density(xc, rho_data, spin):
    rho = _total_rho(rho_data, spin)
    exc = _eval_xc(xc, rho_data, deriv=0, spin=spin)
    return rho * exc


@partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def _xc_potential(xc, rho_data, spin):
    return _eval_xc(xc, rho_data, deriv=1, spin=spin)


@_xc_energy_density.defjvp
def _xc_energy_density_jvp(xc, spin, primals, tangents):

    (rho_data,), (rho_data_dot,) = primals, tangents

    exc_ = _xc_energy_density(xc, rho_data, spin)
    vxc = _xc_potential(xc, rho_data, spin)

    if spin:
        d_exc_ = jnp.einsum("...is,...is->...", vxc, rho_data_dot)
    else:
        d_exc_ = jnp.einsum("...i,...i->...", vxc, rho_data_dot)

    return exc_, d_exc_


@_xc_potential.defjvp
def _xc_potential_jvp(xc, spin, primals, tangents):

    (rho_data,), (rho_data_dot,) = primals, tangents

    vxc = _xc_potential(xc, rho_data, spin)
    fxc = _eval_xc(xc, rho_data, deriv=2, spin=spin)

    if spin:
        d_vxc = jnp.einsum("...isjr,...jr->...is", fxc, rho_data_dot)
    else:
        d_vxc = jnp.einsum("...ij,...j->...i", fxc, rho_data_dot)

    return vxc, d_vxc


def xc_energy_density(
    xc: str,
    rho: ScalarFeature,
    grad_rho: VectorFeature | None = None,
    tau: ScalarFeature | None = None,
    *,
    spin: bool = True,
) -> ScalarFeature:
    rho_data = _stack_density_data(rho, grad_rho, tau, spin=spin)
    return _xc_energy_density(xc, rho_data, spin)


def _deriv_order(xc_type) -> int:

    if xc_type == "HF" or xc_type == "LDA":
        return 0
    elif xc_type == "GGA" or xc_type == "MGGA":
        return 1
    else:
        raise ValueError(f"Unsupported XC type: {xc_type}")


class LibXCEnergyDensity(XCEnergyDensity):
    """Differentiable pointwise energy density backed by PySCF LibXC.

    Args:
        xc_code: Any functional name or LibXC expression accepted by PySCF.
        spin: If ``True``, inputs carry a final two-component spin axis. If ``False``,
            inputs are spin-summed.

    The callable arguments depend on the functional family: density for LDA; density
    and its Cartesian gradient for GGA; and additionally kinetic-energy density for
    meta-GGA. The result is energy per volume, ``n epsilon_xc``, in atomic units.
    LibXC executes through a host callback, with custom JAX derivatives supplied up to
    the orders requested by the surrounding calculation.
    """

    xc_code: str = eqx.field(static=True)
    xc_type: str = eqx.field(static=True)
    spin: bool = eqx.field(static=True)
    deriv_order: int = eqx.field(static=True)

    def __init__(self, xc_code: str, spin: bool = True):
        self.xc_code = libxc.format_xc_code(xc_code)
        self.xc_type = libxc.xc_type(self.xc_code)
        self.spin = spin
        self.deriv_order = _deriv_order(self.xc_type)

    def __call__(self, *args: ScalarFeature | VectorFeature) -> ScalarFeature:
        return xc_energy_density(self.xc_code, *args, spin=self.spin)
