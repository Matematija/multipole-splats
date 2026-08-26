from functools import partial

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float
from pyscf import dft, gto


def _eval_ao(mol, r, deriv=0):

    r_ = np.atleast_2d(r).astype(np.float64)
    ao_vals = dft.numint.eval_ao(mol, r_, deriv=deriv).astype(r.dtype)

    if deriv > 0:
        ao_vals = np.transpose(ao_vals, (1, 0, 2))  # Move the batch axis to the front

    if r.ndim == 1:
        ao_vals = np.squeeze(ao_vals, axis=0)

    return ao_vals


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _ao_value_and_grad(mol, coords):

    out_shape = jax.ShapeDtypeStruct((4, mol.nao), coords.dtype)

    ao_data = jax.pure_callback(
        lambda r: _eval_ao(mol, r, deriv=1), out_shape, coords, vmap_method="broadcast_all"
    )

    return ao_data[0], ao_data[1:]


@_ao_value_and_grad.defjvp
def _(mol, primals, tangents):

    (coords,), (d_coords,) = primals, tangents
    out_shape = jax.ShapeDtypeStruct((10, mol.nao), coords.dtype)

    ao_data = jax.pure_callback(
        lambda r: _eval_ao(mol, r, deriv=2), out_shape, coords, vmap_method="broadcast_all"
    )

    ao_vals, ao_grad, ao_hess_triu = ao_data[0], ao_data[1:4], ao_data[4:]
    d_vals = jnp.einsum("...x,x...->...", d_coords, ao_grad)

    i, j = jnp.triu_indices(3)
    ao_hess = jnp.zeros((3, 3, mol.nao), dtype=ao_hess_triu.dtype)
    ao_hess = ao_hess.at[i, j, :].set(ao_hess_triu).at[j, i, :].set(ao_hess_triu)
    d_grad = jnp.einsum("xyn,y->xn", ao_hess, d_coords)

    return (ao_vals, ao_grad), (d_vals, d_grad)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def eval_ao(mol: gto.Mole, coords: Float[Array, "3"]) -> Float[Array, " nao"]:

    out_shape = jax.ShapeDtypeStruct((mol.nao,), coords.dtype)

    return jax.pure_callback(
        lambda r: _eval_ao(mol, r, deriv=0), out_shape, coords, vmap_method="broadcast_all"
    )


@eval_ao.defjvp
def _(mol, primals, tangents):
    (coords,), (d_coords,) = primals, tangents
    ao_vals, ao_deriv = _ao_value_and_grad(mol, coords)
    pushfwd = jnp.einsum("...x,x...->...", d_coords, ao_deriv)
    return ao_vals, pushfwd


class AtomicOrbitals(eqx.Module):
    mol: gto.Mole = eqx.field(static=True)

    def __call__(self, r: Float[Array, "3"]) -> Float[Array, " nao"]:
        return eval_ao(self.mol, r)

    def __len__(self) -> int:
        return self.mol.nao
