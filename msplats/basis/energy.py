import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Scalar

from ..xc import XCFunctional
from .integrals import Integrals


def hartree_energy(
    cderi: Float[Array, "naux nao nao"], dm: Float[Array, "*spin nao nao"]
) -> Scalar:

    if dm.ndim == 3:
        dm = jnp.sum(dm, axis=0)
    elif dm.ndim != 2:
        raise ValueError(f"Density matrix has to be 2D or 3D, got shape {dm.shape}.")

    c = jnp.einsum("Akl,kl->A", cderi, dm)
    return 0.5 * jnp.dot(c, c)


def exchange_energy(
    cderi: Float[Array, "naux nao nao"], dm: Float[Array, "*spin nao nao"]
) -> Scalar:

    assert dm.ndim in (2, 3), f"Density matrix has to be 2D or 3D, got shape {dm.shape}."
    prefactor = 0.25 if dm.ndim == 2 else 0.5

    aux = jnp.einsum("Amn,...np->...Amp", cderi, dm)
    return -prefactor * jnp.tensordot(aux, aux.mT, axes=aux.ndim)


def J_matrix(cderi: Float[Array, "naux nao nao"], dm: Float[Array, "*spin nao nao"]) -> Scalar:
    return jax.grad(hartree_energy, argnums=1)(cderi, dm)


def K_matrix(cderi: Float[Array, "naux nao nao"], dm: Float[Array, "*spin nao nao"]) -> Scalar:
    assert dm.ndim in (2, 3), f"Density matrix has to be 2D or 3D, got shape {dm.shape}."
    prefactor = 2 if dm.ndim == 2 else 1
    return -prefactor * jax.grad(exchange_energy, argnums=1)(cderi, dm)


class EnergyFunctional(eqx.Module):
    ints: Integrals
    xc: XCFunctional

    def __call__(self, dm: Float[Array, "*spin nao nao"], *args, **kwargs) -> Scalar:

        E0 = self.ints.energy_nuc
        E1 = jnp.tensordot(dm, self.ints.int_1e, axes=2).sum()
        E2 = hartree_energy(self.ints.cderi, dm)
        Exc = self.xc(dm, *args, **kwargs)

        return E0 + E1 + E2 + Exc
