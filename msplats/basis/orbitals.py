from collections.abc import Callable

import equinox as eqx
import jax
import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from jax.nn.initializers import normal
from jaxtyping import Float, Int, Key, Scalar, ScalarLike
from pyscf import gto

from ..utils import eigh, psd_inv_sqrt
from .basis import AtomicOrbitals

_ParamInitFn = Callable[[Key, tuple[int, ...]], Array]

AOMatrix = Float[Array, "s nao nao"] | Float[Array, "nao nao"]
ScalarFeature = Float[Array, "... s"] | Float[Array, "..."]
VectorFeature = Float[Array, "... i s"] | Float[Array, "... i"]


class MolecularOrbitals(eqx.Module):
    coeffs: Float[Array, "... nao nmo"]
    ao: AtomicOrbitals

    def __call__(self, r: Float[Array, "3"]) -> Float[Array, "... nmo"]:
        return jnp.einsum("...mi,m->...i", self.coeffs, self.ao(r))


class LCAO(eqx.Module):
    params: Float[Array, "#spin nao nao"]
    ao: AtomicOrbitals
    inv_sqrt_ovlp: Float[Scalar, "nao nao"]

    def __init__(
        self,
        ao: AtomicOrbitals,
        overlap: Float[Array, "nao nao"],
        spin: bool = True,
        init_fn: _ParamInitFn | None = None,
        *,
        key: Key,
    ):

        nao = len(ao)
        self.ao = ao

        if init_fn is None:
            init_fn = normal(stddev=1e-3)

        self.params = init_fn(key, (2, nao, nao) if spin else (nao, nao))
        self.inv_sqrt_ovlp = psd_inv_sqrt(overlap, rcond=1e-5)

    def coeffs(self):
        Q, _ = jnp.linalg.qr(self.params)
        inv_sqrt_ovlp = lax.stop_gradient(self.inv_sqrt_ovlp)
        return inv_sqrt_ovlp @ Q

    def __call__(self, r: Float[Array, "3"]) -> Float[Array, "#spin nao"]:
        return jnp.einsum("smi,...m->s...i", self.coeffs(), self.ao(r))


class SingleElectronProblem(eqx.Module):
    """Solve an AO-basis generalized one-electron eigenproblem.

    Args:
        integrals_1e: Fixed Hamiltonian matrix ``h_1``, optionally with a leading spin
            axis.
        overlap: AO overlap matrix ``S`` using the same spin convention.
        grad_eps: Energy-gap cutoff below which eigenvector derivative couplings are
            suppressed for numerical stability.
        inv_rcond: Relative cutoff used to form ``S^{-1/2}``. The default scales
            machine precision by the matrix size.

    Calling the instance with an additive AO potential matrix ``V`` solves
    ``(h_1 + V) C = S C epsilon`` and returns ordered orbital energies and
    ``S``-orthonormal AO coefficients. Restricted matrices have shape ``(nao, nao)``;
    spin-dependent matrices may carry a leading spin axis.
    """

    h1: AOMatrix
    inv_sqrt_ovlp: AOMatrix
    eps: Scalar

    def __init__(
        self,
        integrals_1e: AOMatrix,
        overlap: AOMatrix,
        grad_eps: ScalarLike = 1e-6,
        inv_rcond: float | None = None,
    ):

        if inv_rcond is None:
            inv_rcond_ = jnp.finfo(overlap.dtype).eps * max(overlap.shape)
        else:
            inv_rcond_ = inv_rcond

        self.h1 = integrals_1e
        self.inv_sqrt_ovlp = psd_inv_sqrt(overlap, rcond=inv_rcond_)
        self.eps = jnp.asarray(grad_eps, dtype=overlap.dtype)

    def __call__(self, v_mat: AOMatrix) -> tuple[Float[Array, " nao"], AOMatrix]:
        hamiltonian = self.inv_sqrt_ovlp @ (self.h1 + v_mat) @ self.inv_sqrt_ovlp
        energies, coeff = eigh(hamiltonian, eps=self.eps)
        # energies, coeff = jax.scipy.linalg.eigh(hamiltonian)
        return energies, self.inv_sqrt_ovlp @ coeff


class Density(eqx.Module):
    """Evaluate an AO density matrix in real space.

    Args:
        dm: Restricted spin-summed density matrix of shape ``(nao, nao)`` or
            unrestricted matrix of shape ``(2, nao, nao)`` with spin first.
        ao: Atomic-orbital evaluator using the same molecule and AO basis as ``dm``.

    Calling the instance evaluates ``n(r) = sum_mn dm_mn chi_m(r) chi_n(r)``. An
    unrestricted input retains its leading spin components. ``from_pyscf`` is a
    convenience constructor that creates the matching :class:`AtomicOrbitals` object.
    """

    dm: AOMatrix = eqx.field(converter=jnp.asarray)
    ao: AtomicOrbitals

    @classmethod
    def from_pyscf(cls, mol: gto.Mole, dm: AOMatrix):
        dm = jnp.asarray(dm)
        return cls(dm, AtomicOrbitals(mol))

    def __post_init__(self):
        if self.dm.ndim not in (2, 3):
            raise ValueError(f"Density matrix has to be 2D or 3D, got shape {self.dm.shape}.")

    def __call__(self, r: Float[Array, "... 3"]) -> Float[Array, "..."]:
        ao_vals = self.ao(r)
        return jnp.einsum("...mn,...m,...n->...", self.dm, ao_vals, ao_vals)


def _coulomb_mat_pyscf(mol, r):
    r = np.asarray(r, dtype=np.float64)
    return mol.intor("int1e_grids", grids=r.reshape(-1, 3))


def _grad_coulomb_mat_pyscf(mol, r):
    r = np.asarray(r, dtype=np.float64)
    return mol.intor("int1e_grids_ip", grids=r.reshape(-1, 3))


def _coulomb_pot_pyscf(mol, dm, r):
    dm = np.asarray(dm, dtype=np.float64)
    mat = _coulomb_mat_pyscf(mol, r)
    v_val = np.einsum("smn,...mn->s...", mat, dm)
    # return v_val if v_val.shape[0] > 1 else v_val.squeeze(0)
    return np.squeeze(v_val)


def _grad_coulomb_pot_pyscf(mol, dm, r):
    dm = np.asarray(dm, dtype=np.float64)
    grad_mat = _grad_coulomb_mat_pyscf(mol, r)
    grad_v_val = np.einsum("csmn,...mn->...sc", grad_mat, dm)
    # return grad_v_val if grad_v_val.shape[-2] > 1 else grad_v_val.squeeze(-2)
    return np.squeeze(grad_v_val)


@eqx.filter_custom_vjp
def _coulomb_potential(diff_args, mol):

    dm, r = diff_args
    result_shape_dtypes = jax.ShapeDtypeStruct(dm.shape[:-2], r.dtype)

    return eqx.filter_pure_callback(
        _coulomb_pot_pyscf,
        mol,
        dm,
        r,
        result_shape_dtypes=result_shape_dtypes,
        vmap_method="expand_dims",
    )


@_coulomb_potential.def_fwd
def _coulomb_potential_fwd(perturbed, diff_args, mol):

    dm_perturbed, _ = perturbed

    if dm_perturbed:
        raise NotImplementedError("Coulomb potential VJP w.r.t. density matrix not implemented.")

    return _coulomb_potential(diff_args, mol), None


@_coulomb_potential.def_bwd
def _coulomb_potential_bwd(_, grads, __, diff_args, mol):

    dm, r = diff_args
    grad_shape_dtypes = jax.ShapeDtypeStruct(dm.shape[:-2] + (3,), r.dtype)

    grad_v_val = eqx.filter_pure_callback(
        _grad_coulomb_pot_pyscf,
        mol,
        dm,
        r,
        result_shape_dtypes=grad_shape_dtypes,
        vmap_method="expand_dims",
    )

    d_r = jnp.tensordot(grads, grad_v_val, axes=grads.ndim)

    return None, d_r


class HartreePotential(eqx.Module):
    """Evaluate the Coulomb potential generated by an AO density matrix.

    Args:
        mol: PySCF molecule defining the AO basis; it is static under JAX transforms.
        dm: AO density matrix with optional leading spin axes.

    Calling the instance at a Bohr-coordinate point returns
    ``integral n(r') / |r-r'| dr'``. Coordinate derivatives are supported, but
    differentiation with respect to the density matrix is not currently implemented.
    """

    mol: gto.Mole = eqx.field(static=True)
    dm: Float[Array, "... nao nao"] = eqx.field(converter=jnp.asarray)

    def __call__(self, r: Float[Array, "3"]) -> Float[Array, "..."]:
        return _coulomb_potential((self.dm, r), self.mol)


class NuclearPotential(eqx.Module):
    atom_coords: Float[Array, "n_atoms 3"] = eqx.field(converter=jnp.asarray)
    atom_charges: Int[Array, " n_atoms"] = eqx.field(converter=jnp.asarray)
    regularize: bool | float = True

    def __call__(self, r: Float[Array, "3"]) -> Scalar:

        if isinstance(self.regularize, float):
            eps = self.regularize
        elif self.regularize:
            eps = jnp.finfo(r.dtype).eps
        else:
            eps = 0.0

        diffs = r - self.atom_coords
        d2 = jnp.sum(diffs**2, axis=-1)

        return self.atom_charges @ lax.rsqrt(d2 + eps**2)
