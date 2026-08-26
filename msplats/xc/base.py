import abc
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Scalar

from ..basis import AtomicOrbitals
from ..grid import Grid


class XCFunctional(eqx.Module):
    name: eqx.AbstractVar[str]
    xc_type: eqx.AbstractVar[str]

    @abc.abstractmethod
    def __call__(self, dm: Float[Array, "*spin nao nao"]) -> Scalar:
        raise NotImplementedError


ScalarFeature = Float[Array, "s"] | Scalar
VectorFeature = Float[Array, "i s"] | Float[Array, "i"]


class XCEnergyDensity(eqx.Module):
    xc_code: eqx.AbstractVar[str]
    xc_type: eqx.AbstractVar[str]
    deriv_order: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(self, *args: ScalarFeature | VectorFeature) -> ScalarFeature:
        raise NotImplementedError


def _density(dm, ao_vals):

    if dm.ndim == 2:
        contraction = "mn,...m,...n->..."
    elif dm.ndim == 3:
        contraction = "smn,...m,...n->...s"
    else:
        raise ValueError(f"Density matrix has to be 2D or 3D, got shape {dm.shape}.")

    return jnp.einsum(contraction, dm, ao_vals, ao_vals)


def _grad_density(dm, ao_vals, ao_grads):

    if dm.ndim == 2:
        contraction = "mn,...m,...ni->...i"
    elif dm.ndim == 3:
        contraction = "smn,...m,...ni->...is"
    else:
        raise ValueError(f"Density matrix has to be 2D or 3D, got shape {dm.shape}.")

    return 2 * jnp.einsum(contraction, dm, ao_vals, ao_grads)


def _kinetic_density(dm, ao_grads):

    if dm.ndim == 2:
        contraction = "mn,...mi,...ni->..."
    elif dm.ndim == 3:
        contraction = "smn,...mi,...ni->...s"
    else:
        raise ValueError(f"Density matrix has to be 2D or 3D, got shape {dm.shape}.")

    return jnp.einsum(contraction, dm, ao_grads, ao_grads)


class _AOData(NamedTuple):
    vals: Float[Array, "ngrid nao"]
    grads: Float[Array, "ngrid nao 3"] | None = None


def _eval_ao_data(xc_type, ao, grid):

    vals = jax.vmap(ao)(grid.coords)

    if xc_type == "LDA":
        return _AOData(vals)
    elif xc_type in ("GGA", "MGGA"):
        grads: Float[Array, "ngrid nao 3"] = jax.vmap(eqx.filter_jacfwd(ao))(grid.coords)  # pyright: ignore
        return _AOData(vals, grads)
    else:
        raise ValueError(f"Unsupported local functional xc_type: {xc_type}")


class LocalFunctional(XCFunctional):
    functional: XCEnergyDensity
    ao: AtomicOrbitals
    grid: Grid

    ao_cache: _AOData | None = eqx.field(repr=False)

    def __init__(
        self, functional: XCEnergyDensity, ao: AtomicOrbitals, grid: Grid, *, cache_ao: bool = True
    ):
        self.xc_type = functional.xc_type
        self.name = functional.xc_code
        self.functional = functional
        self.ao = ao
        self.grid = grid

        if cache_ao:
            self.ao_cache = _eval_ao_data(self.xc_type, ao, grid)
        else:
            self.ao_cache = None

    def _prepare_local_inputs(self, dm, ao_data):

        args = [_density(dm, ao_data.vals)]

        if self.functional.xc_type in ("GGA", "MGGA"):
            grad_density_vals = _grad_density(dm, ao_data.vals, ao_data.grads)
            args.append(grad_density_vals)

        if self.functional.xc_type == "MGGA":
            tau_vals = _kinetic_density(dm, ao_data.grads)
            args.append(tau_vals)

        return args

    def __call__(self, dm: Float[Array, "*spin nao nao"], *ao_eval: Float[Array, "..."]) -> Scalar:

        if ao_eval:
            ao_data = _AOData(*ao_eval)
        elif self.ao_cache is not None:
            ao_data = self.ao_cache
        else:
            ao_data = _eval_ao_data(self.xc_type, self.ao, self.grid)

        args = self._prepare_local_inputs(dm, ao_data)
        exc = jax.vmap(lambda a: self.functional(*a))(args)

        return self.grid.weights @ exc


class LocalPotential(eqx.Module):
    functional: XCEnergyDensity
    density: Callable[[Float[Array, "3"]], Scalar]

    def __call__(self, r: Float[Array, "3"]) -> Scalar:

        if self.functional.deriv_order > 1:
            raise NotImplementedError("Higher-order derivatives not implemented.")

        if self.functional.deriv_order == 0:
            return eqx.filter_grad(self.functional)(self.density(r))

        # self.functional.deriv_order == 1

        def aux(r):
            rho, grad_rho = eqx.filter_value_and_grad(self.density)(r)
            dn, dgn = jax.grad(self.functional, argnums=(0, 1))(rho, grad_rho)
            return dgn, dn

        J, dn = jax.jacrev(aux, has_aux=True)(r)
        return dn - jnp.trace(J)
