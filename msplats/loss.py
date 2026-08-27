from abc import abstractmethod

import equinox as eqx
import jax
from jax import Array, lax
from jax import numpy as jnp
from jaxtyping import Float, Scalar

from .basis import AtomicOrbitals, EnergyFunctional, J_matrix, SingleElectronProblem
from .grid import Grid
from .potential import MultipolePotential
from .utils import reduce, vmap


class _GridCache(eqx.Module):
    ao_vals: Float[Array, "n_grid nao"]
    density_vals: Float[Array, " n_grid"]

    def __init__(
        self,
        orb_basis: AtomicOrbitals,
        grid: Grid,
        dm_ref: Float[Array, "*spin nao nao"],
    ):

        self.ao_vals = jax.vmap(orb_basis)(grid.coords)
        self.density_vals = jnp.einsum("...mn,xm,xn->x", dm_ref, self.ao_vals, self.ao_vals)


class _GridData(eqx.Module):
    orb_basis: AtomicOrbitals
    grid: Grid
    dm_ref: Float[Array, "*spin nao nao"]
    cache: _GridCache | None = eqx.field(repr=False)
    chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        orb_basis: AtomicOrbitals,
        grid: Grid,
        dm_ref: Float[Array, "*spin nao nao"],
        cache_grid_data: bool,
        chunk_size: int | None,
    ):

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"Chunk size has to be positive, got {chunk_size}.")

        self.orb_basis = orb_basis
        self.grid = grid
        self.dm_ref = jnp.asarray(dm_ref)
        self.chunk_size = chunk_size
        self.cache = _GridCache(orb_basis, grid, self.dm_ref) if cache_grid_data else None

    def potential_values(self, potential: MultipolePotential) -> Float[Array, " n_grid"]:
        return vmap(potential, chunk_size=self.chunk_size)(self.grid.coords)

    def potential_matrix(self, v_vals: Float[Array, " n_grid"]) -> Float[Array, "nao nao"]:

        if self.cache is not None:
            ao_vals = lax.stop_gradient(self.cache.ao_vals)
            return jnp.einsum("x,x,xm,xn->mn", self.grid.weights, v_vals, ao_vals, ao_vals)

        nao = len(self.orb_basis)
        dtype = jnp.result_type(self.grid.weights, v_vals)
        init_value = jnp.zeros((nao, nao), dtype=dtype)

        def contribution(coords, weights, values):
            ao_vals = jax.vmap(self.orb_basis)(coords)
            return jnp.einsum("x,x,xm,xn->mn", weights, values, ao_vals, ao_vals)

        return reduce(
            contribution,
            self.grid.coords,
            self.grid.weights,
            v_vals,
            init_value=init_value,
            chunk_size=self.chunk_size,
        )

    def density_values(self, dm: Float[Array, "*spin nao nao"]) -> Float[Array, " n_grid"]:

        if self.cache is not None:
            ao_vals = lax.stop_gradient(self.cache.ao_vals)
            return jnp.einsum("...mn,xm,xn->x", dm, ao_vals, ao_vals)

        def density(r):
            ao_vals = self.orb_basis(r)
            return jnp.einsum("...mn,m,n->", dm, ao_vals, ao_vals)

        return vmap(density, chunk_size=self.chunk_size)(self.grid.coords)

    def reference_density_values(self) -> Float[Array, " n_grid"]:

        if self.cache is not None:
            return lax.stop_gradient(self.cache.density_vals)

        return self.density_values(lax.stop_gradient(self.dm_ref))

    def regularization(self, potential: MultipolePotential) -> Scalar:
        """Force matching against the fixed background, leaving only the splat field."""

        value_and_grad = eqx.filter_value_and_grad(potential)

        def contribution(coords, weights):
            _, grad_vals = jax.vmap(value_and_grad)(coords)
            return jnp.einsum("x,xi,xi->", weights, grad_vals, grad_vals)

        integral = reduce(
            contribution,
            self.grid.coords,
            self.grid.weights,
            init_value=jnp.zeros((), dtype=self.grid.weights.dtype),
            chunk_size=self.chunk_size,
        )

        return integral / (8 * jnp.pi)


class _OrbitalProblem(eqx.Module):
    grid_data: _GridData
    eigensolver: SingleElectronProblem
    mo_occ: Float[Array, "*spin nao"]

    def __init__(
        self,
        orb_basis: AtomicOrbitals,
        energy: EnergyFunctional,
        grid: Grid,
        dm_ref: Float[Array, "*spin nao nao"],
        mo_occ: Float[Array, "*spin nao"] | None,
        cache_grid_data: bool,
        chunk_size: int | None,
        eigh_grad_eps: float,
    ):

        dm_ref = jnp.asarray(dm_ref)
        nao = len(orb_basis)

        if dm_ref.ndim not in (2, 3) or dm_ref.shape[-2:] != (nao, nao):
            raise ValueError(
                f"Reference density matrix has to end in shape ({nao}, {nao}), got {dm_ref.shape}."
            )

        electron_count = jnp.sum(energy.ints.ovlp * dm_ref)
        n_electrons = int(jnp.rint(electron_count).item())

        if n_electrons <= 0:
            raise ValueError(f"Electron count has to be positive, got {n_electrons}.")
        if not bool(jnp.isclose(electron_count, n_electrons, atol=1e-5, rtol=1e-5).item()):
            raise ValueError(f"Reference density has non-integral electron count {electron_count}.")

        if mo_occ is None:
            if dm_ref.ndim != 2 or n_electrons % 2:
                raise ValueError(
                    "Explicit occupations are required for unrestricted or open-shell densities."
                )
            if n_electrons // 2 > nao:
                raise ValueError(
                    f"Cannot place {n_electrons} electrons in {nao} restricted orbitals."
                )

            mo_occ_ = jnp.zeros(nao, dtype=dm_ref.dtype)
            mo_occ_ = mo_occ_.at[: n_electrons // 2].set(2.0)
        else:
            mo_occ_ = jnp.asarray(mo_occ, dtype=dm_ref.dtype)

            if mo_occ_.ndim != dm_ref.ndim - 1 or mo_occ_.shape[-1] != nao:
                raise ValueError(
                    "Occupation and reference-density spin conventions disagree: "
                    f"got shapes {mo_occ_.shape} and {dm_ref.shape}."
                )
            if not bool(jnp.isclose(jnp.sum(mo_occ_), n_electrons, atol=1e-5, rtol=1e-5).item()):
                raise ValueError(f"Occupations sum to {jnp.sum(mo_occ_)}, expected {n_electrons}.")

        J_fa = ((n_electrons - 1) / n_electrons) * J_matrix(energy.ints.cderi, dm_ref)

        self.grid_data = _GridData(
            orb_basis, grid, dm_ref, cache_grid_data=cache_grid_data, chunk_size=chunk_size
        )
        self.eigensolver = SingleElectronProblem(
            energy.ints.int_1e + J_fa, energy.ints.ovlp, grad_eps=eigh_grad_eps
        )
        self.mo_occ = mo_occ_

    def density_matrix(
        self, mo_coeff: Float[Array, "*spin nao nao"]
    ) -> Float[Array, "*spin nao nao"]:
        return jnp.einsum("...i,...mi,...ni->...mn", self.mo_occ, mo_coeff, mo_coeff)


class AbstractLoss(eqx.Module):
    """Abstract scalar objective over a variational multipole potential.

    Concrete losses are Equinox modules whose only call argument is a
    :class:`MultipolePotential`. The result is a scalar to minimize and can be
    differentiated with ``eqx.filter_value_and_grad``.
    """

    @abstractmethod
    def __call__(self, potential: MultipolePotential) -> Scalar:
        raise NotImplementedError


class _OrbitalLoss(AbstractLoss):
    _problem: _OrbitalProblem


class EnergyLoss(_OrbitalLoss):
    """Generalized Kohn--Sham energy induced by a local multipole potential.

    Args:
        orb_basis: Atomic-orbital evaluator for the molecular AO basis.
        energy: Orbital-dependent energy functional to minimize.
        grid: Quadrature used to project the variational potential into the AO basis.
        dm_ref: Reference AO density matrix used to construct the fixed Fermi--Amaldi
            background and determine the electron count.
        mo_occ: Orbital occupations. Closed-shell restricted occupations are inferred
            when omitted; unrestricted and open-shell calculations must provide them.
        cache_grid_data: Materialize all AO values and the reference density on the
            grid. If ``False``, evaluate AO data lazily in chunks.
        chunk_size: Number of grid points processed per chunk, or ``None`` for one
            unchunked operation.
        eigh_grad_eps: Degeneracy cutoff for derivatives of orbital eigenvectors.

    The trial Hamiltonian contains the nuclear one-electron operator, a fixed
    Fermi--Amaldi term built from ``dm_ref``, and the supplied ``v_MS``. Calling the
    loss solves for its occupied orbitals and returns the full energy from ``energy``
    in Hartree. All molecular objects must share geometry, AO basis, and spin
    convention.
    """

    energy: EnergyFunctional

    def __init__(
        self,
        orb_basis: AtomicOrbitals,
        energy: EnergyFunctional,
        grid: Grid,
        dm_ref: Float[Array, "*spin nao nao"],
        mo_occ: Float[Array, "*spin nao"] | None = None,
        *,
        cache_grid_data: bool = True,
        chunk_size: int | None = 256,
        eigh_grad_eps: float = 1e-6,
    ):

        self.energy = energy
        self._problem = _OrbitalProblem(
            orb_basis,
            energy,
            grid,
            dm_ref,
            mo_occ,
            cache_grid_data,
            chunk_size,
            eigh_grad_eps,
        )

    def __call__(self, potential: MultipolePotential) -> Scalar:

        v_vals = self._problem.grid_data.potential_values(potential)
        v_mat = self._problem.grid_data.potential_matrix(v_vals)
        _, mo_coeff = self._problem.eigensolver(v_mat)
        dm = self._problem.density_matrix(mo_coeff)

        return self.energy(dm)


def _wu_yang_energy(v_vals, ks_prob, mo_occ, grid_data):

    v_mat = grid_data.potential_matrix(v_vals)
    _, mo_coeff = ks_prob(v_mat)

    dm = jnp.einsum("...i,...mi,...ni->...mn", mo_occ, mo_coeff, mo_coeff)
    E1 = jnp.tensordot(dm, ks_prob.h1, axes=2).sum()

    rho_vals = grid_data.density_values(dm)
    rho_ref_vals = grid_data.reference_density_values()
    wrho_diff = grid_data.grid.weights * (rho_vals - rho_ref_vals)
    E2 = wrho_diff @ v_vals

    return E1 + E2, wrho_diff


@eqx.filter_custom_vjp
def _wu_yang_aux(v_vals, ks_prob, mo_occ, grid_data):
    W, wrho_diff = _wu_yang_energy(v_vals, ks_prob, mo_occ, grid_data)
    err = jnp.sum(jnp.abs(wrho_diff))
    return W, err


@_wu_yang_aux.def_fwd
def _wu_yang_aux_fwd(_, v_vals, ks_prob, mo_occ, grid_data):
    W, wrho_diff = _wu_yang_energy(v_vals, ks_prob, mo_occ, grid_data)
    err = jnp.sum(jnp.abs(wrho_diff))
    return (W, err), wrho_diff


@_wu_yang_aux.def_bwd
def _wu_yang_aux_bwd(wrho_diff, cotangents, _, v_vals, ks_prob, mo_occ, grid_data):
    dW, derr = cotangents
    assert derr is None, f"Error term should not have a cotangent, got {derr}"
    return dW * wrho_diff


class WuYangLoss(_OrbitalLoss):
    """Negative Wu--Yang functional for inverse Kohn--Sham optimization.

    Args:
        orb_basis: Atomic-orbital evaluator for the molecular AO basis.
        energy: Energy object supplying molecular integrals. Its XC functional is not
            evaluated by the Wu--Yang objective.
        grid: Quadrature used for potential projection and density comparison.
        dm_ref: Target restricted or unrestricted AO density matrix.
        mo_occ: Orbital occupations. Closed-shell restricted occupations are inferred
            when omitted; unrestricted and open-shell calculations must provide them.
        cache_grid_data: Materialize AO and reference-density values on the grid. If
            ``False``, evaluate them lazily in chunks.
        chunk_size: Number of grid points processed per chunk, or ``None`` for one
            unchunked operation.
        eigh_grad_eps: Degeneracy cutoff for derivatives of orbital eigenvectors.

    Calling the loss returns ``-W[v_MS]`` so ordinary minimizers maximize the
    Wu--Yang functional. Its custom reverse-mode rule uses the weighted density
    residual directly and does not form or invert the Kohn--Sham response. The fixed
    one-electron Hamiltonian includes the nuclear and Fermi--Amaldi terms.
    """

    def __init__(
        self,
        orb_basis: AtomicOrbitals,
        energy: EnergyFunctional,
        grid: Grid,
        dm_ref: Float[Array, "*spin nao nao"],
        mo_occ: Float[Array, "*spin nao"] | None = None,
        *,
        cache_grid_data: bool = True,
        chunk_size: int | None = 256,
        eigh_grad_eps: float = 1e-6,
    ):

        self._problem = _OrbitalProblem(
            orb_basis,
            energy,
            grid,
            dm_ref,
            mo_occ,
            cache_grid_data,
            chunk_size,
            eigh_grad_eps,
        )

    def __call__(self, potential: MultipolePotential) -> Scalar:

        v_vals = self._problem.grid_data.potential_values(potential)
        W, _ = _wu_yang_aux(
            v_vals,
            self._problem.eigensolver,
            self._problem.mo_occ,
            self._problem.grid_data,
        )

        return -W


class Regularized(AbstractLoss):
    """Add force-matching regularization to an OEP or IKS loss.

    Args:
        loss: :class:`EnergyLoss` or :class:`WuYangLoss` to augment.
        lambda_: Nonnegative regularization strength.

    The returned objective is ``loss(v_MS) + lambda_ R[v_MS]``, with
    ``R = (8 pi)^{-1} integral |grad v_MS(r)|^2 dr`` on the wrapped loss's grid.
    This selects a smoother representative among nearly degenerate potentials; it is
    not required for convergence and does not remove the physical response null space.
    """

    loss: _OrbitalLoss
    lambda_: float = eqx.field(static=True)

    def __init__(self, loss: EnergyLoss | WuYangLoss, lambda_: float):

        if lambda_ < 0:
            raise ValueError(f"Regularization strength has to be nonnegative, got {lambda_}.")

        self.loss = loss
        self.lambda_ = float(lambda_)

    def __call__(self, potential: MultipolePotential) -> Scalar:

        reg = self.loss._problem.grid_data.regularization(potential)
        return self.loss(potential) + self.lambda_ * reg


__all__ = ["AbstractLoss", "EnergyLoss", "Regularized", "WuYangLoss"]
