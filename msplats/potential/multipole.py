from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import erf, logit
from jaxtyping import Float, Int, Key, Scalar
from pyscf import df, gto, lo

from ..data import COVALENT, VDW
from ..utils import reduce

PySCFMolecule = gto.Mole
Activation = Callable[[Float[Array, "..."]], Float[Array, "..."]]


class GeometryInitializer(eqx.Module):
    @abstractmethod
    def __call__(
        self, num_points: int, key: Key
    ) -> tuple[Float[Array, "num_points 3"], Float[Array, " num_points"]]:
        """
        Generate initial positions and log-exponents for the multipole potential.

        Args:
            num_points: Total number of multipole points to generate.
            key: JAX PRNG key for random number generation.

        Returns:
            positions: Array of shape (num_points, 3) with initial positions.
            log_exponents: Array of shape (num_points,) with initial log-exponents.
        """
        raise NotImplementedError


class AtomGaussianInitializer(GeometryInitializer):
    atom_coords: Float[Array, "n_atoms 3"] = eqx.field(converter=jnp.asarray)
    atom_charges: Int[Array, " n_atoms"] = eqx.field(converter=jnp.asarray)
    position_scale: Scalar = eqx.field(default=0.4, converter=jnp.asarray)
    exponent_bounds: Float[Array, "2"] = eqx.field(default=(1e-2, 1e3), converter=jnp.asarray)

    def __call__(
        self, num_points: int, key: Key
    ) -> tuple[Float[Array, "num_points 3"], Float[Array, " num_points"]]:

        n_atoms = len(self.atom_charges)
        atom_key, gaussian_key, exponent_key = jr.split(key, 3)

        atom_idx = jr.choice(atom_key, n_atoms, shape=(num_points,))
        mean = self.atom_coords[atom_idx]
        std = self.position_scale * VDW[self.atom_charges[atom_idx]]
        z = jr.normal(gaussian_key, shape=(num_points, 3))
        positions = mean + z * std[:, None]

        log_min, log_max = jnp.log(jnp.sort(self.exponent_bounds))
        u = jr.uniform(exponent_key, shape=(num_points,))
        log_exponents = log_min + (log_max - log_min) * u

        return positions, log_exponents


_PySCFMeanField = Any


def _run_boys(mf):
    def localize(C) -> np.ndarray:
        boys = lo.Boys(mf.mol, C)
        boys.verbose = 0
        return boys.kernel()  # pyright: ignore

    if mf.mo_coeff.ndim == 3:  # Unrestricted
        (Ca, Cb), (na, nb) = mf.mo_coeff, mf.mo_occ
        Ca_loc = localize(Ca[:, na > 0])
        Cb_loc = localize(Cb[:, nb > 0])
        C_loc = np.concatenate([Ca_loc, Cb_loc], axis=1)
    else:
        C, n = mf.mo_coeff, mf.mo_occ
        C_loc = localize(C[:, n > 0])

    return jnp.asarray(C_loc)


class BoysInitializer(GeometryInitializer):
    means: Float[Array, "n_points 3"]
    vars: Float[Array, "n_points 3"]
    exponent_spread: float

    def __init__(
        self, mf: _PySCFMeanField, exponent_spread: float = 1.0, min_length_scale: float = 1e-2
    ):

        if not mf.converged:
            raise ValueError(
                "SCF calculation did not converge. Please provide a converged mean-field object."
            )

        C_loc = _run_boys(mf)
        ao_dip = jnp.asarray(mf.mol.intor("int1e_r", comp=3))
        # ao_r2 = jnp.asarray(mf.mol.intor("int1e_r2", comp=3))

        ao_r2 = mf.mol.intor("int1e_rr").reshape(3, 3, mf.mol.nao, mf.mol.nao)
        ao_r2 = jnp.stack([ao_r2[i, i] for i in range(3)], axis=0)

        r_avg = jnp.einsum("imn,ma,na->ai", ao_dip, C_loc, C_loc)
        r2_avg = jnp.einsum("imn,ma,na->ai", ao_r2, C_loc, C_loc)
        variances = r2_avg - r_avg**2

        idx = jnp.argsort(variances.sum(axis=1))
        self.means = r_avg[idx, :]
        self.vars = jnp.maximum(variances[idx], min_length_scale**2)
        self.exponent_spread = exponent_spread

    def __call__(
        self, num_points: int, key: Key
    ) -> tuple[Float[Array, "num_points 3"], Float[Array, " num_points"]]:

        n_lmo = len(self.means)
        pos_key, exp_key = jax.random.split(key)

        # 1. Distribute points
        base_count, remainder = divmod(num_points, n_lmo)
        counts = jnp.full(n_lmo, base_count).at[:remainder].add(1)

        # 2. Expand stats
        lmo_idx = jnp.repeat(jnp.arange(n_lmo), counts)  # (N, 3)
        var_diag = self.vars[lmo_idx]  # (N, 3)

        # 3. Sample Positions (Anisotropic)
        # The noise is scaled by the per-axis standard deviation.
        # This creates a cloud shaped like the orbital (e.g., cigar-shaped for bonds)
        z = jax.random.normal(pos_key, (num_points, 3))
        positions = self.means[lmo_idx] + z * jnp.sqrt(var_diag)

        # 4. Sample Exponents
        # Geometric mean of variances gives an "effective volume"
        # var_eff = (var_x * var_y * var_z)^(1/3)
        # alpha_base ~ 1 / (2 * var_eff)
        # This is a heuristic: it picks an exponent that matches the "average" width.
        # (Since we use spherical Gaussians, we can't perfectly match the ellipsoid,
        # but this centers the exponent distribution correctly).

        var_eff = jnp.prod(var_diag, axis=-1) ** (1 / 3)
        alpha_base = 1 / (2 * var_eff)

        log_alpha_mean = jnp.log(alpha_base)
        z = jax.random.normal(exp_key, (num_points,))
        log_exponents = log_alpha_mean + self.exponent_spread * z

        return positions, log_exponents


def _to_exponent_logits(log_exponents, log_bounds, eps=1e-6):
    log_min, log_max = jnp.unstack(jnp.sort(log_bounds, axis=-1), axis=-1)
    p = (log_exponents - log_min) / (log_max - log_min)
    return logit(p.clip(eps, 1 - eps))


def _nearest_neighbor_distance(coords):
    dists = jnp.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return dists.at[jnp.diag_indices_from(dists)].set(jnp.inf).min(axis=-1)


def _guess_exponent_bounds(atom_charges, atom_coords):
    """
    Default (alpha_min, alpha_max) for the splat widths, from the nuclear frame alone.

    A splat's potential erf(sqrt(alpha) r)/r varies on a length scale l ~ 1/sqrt(alpha),
    so the bounds are set by the sharpest and broadest features the potential should be
    able to resolve:

      * alpha_max: the sharpest physical structure near a nucleus is the 1s cusp, which
        decays on 1/(2Z). Using a charge-weighted effective nucleus Zeff gives
        alpha_max = (2 Zeff)^2. Anything sharper is not physics -- it is the freedom that
        lets the optimizer build delta-like spikes in the response null space.

      * alpha_min: the broadest useful feature is the size of the system. The
        charge-weighted nuclear spread alone is *degenerate* for a single atom (variance
        is identically zero), which would send alpha_min -> infinity and inverted the
        bounds, so it is floored by the largest van der Waals radius: for an atom, its
        own electron cloud is the length scale.

    Returned sorted (min, max); AtomGaussianSplat also sorts on construction.
    """
    Z = atom_charges.astype(float)
    p = Z / Z.sum()

    Zeff = p @ Z
    alpha_max = (2 * Zeff) ** 2

    mean = p @ atom_coords
    var = p @ (atom_coords - mean) ** 2
    spread = 2 * jnp.sqrt(var.sum())  # == the bond length for a homonuclear diatomic
    length_scale = jnp.maximum(spread, VDW[atom_charges].max())
    alpha_min = 1 / (2 * length_scale**2)

    return jnp.sort(jnp.stack([alpha_min, alpha_max], axis=-1), axis=-1)


class AtomGaussianSplat(eqx.Module):
    atom_coords: Float[Array, "n_atoms 3"]
    snap_exponents: Float[Array, " n_atoms"]
    tube_radius: Scalar
    log_exponent_bounds: Float[Array, "n_points 2"]

    position_params: Float[Array, "n_points 3"]
    exponent_params: Float[Array, "n_points"]

    def __init__(
        self,
        num_points: int,
        atom_coords: Float[Array, "n_atoms 3"],
        atom_charges: Int[Array, " n_atoms"],
        geometry_init: GeometryInitializer,
        exponent_bounds: tuple[float, float] | None = None,
        tube_scale: float = 0.6,
        snap_scale: float = 0.5,
        *,
        key: Key,
    ):

        self.atom_coords = atom_coords
        covalent_radii = jnp.asarray(COVALENT[atom_charges])

        if len(atom_charges) == 1:
            nn_dist = covalent_radii
        else:
            nn_dist = _nearest_neighbor_distance(atom_coords)

        snap_radius = jnp.maximum(covalent_radii, 0.5 * nn_dist)
        self.snap_exponents = snap_scale / snap_radius**2
        self.tube_radius = tube_scale * jnp.max(nn_dist)

        if exponent_bounds is None:
            exponent_bounds_ = _guess_exponent_bounds(atom_charges, atom_coords)
        else:
            exponent_bounds_ = jnp.asarray(exponent_bounds)

        # Sort once here so the (min, max) convention is consistent everywhere: the
        # `exponents` property unpacks these positionally and does NOT sort, while
        # _to_exponent_logits does -- unordered bounds would make the two disagree.
        self.log_exponent_bounds = jnp.log(jnp.sort(exponent_bounds_, axis=-1))
        self.position_params, log_exponents = geometry_init(num_points, key)
        self.exponent_params = _to_exponent_logits(log_exponents, self.log_exponent_bounds)

    @property
    def n_points(self) -> int:
        return len(self.exponent_params)

    @property
    def atom_affinity_scores(self):

        anchor_coords = lax.stop_gradient(self.atom_coords)
        snap_exponents = lax.stop_gradient(self.snap_exponents)

        d = self.position_params[:, None, :] - anchor_coords[None, :, :]
        logits = -snap_exponents * jnp.sum(d**2, axis=-1)
        return jax.nn.softmax(logits, axis=-1)

    @property
    def positions(self) -> Float[Array, "n_points 3"]:

        anchor_coords = lax.stop_gradient(self.atom_coords)
        tube_radius = lax.stop_gradient(self.tube_radius)

        snap_anchors = self.atom_affinity_scores @ anchor_coords
        offset = self.position_params - snap_anchors
        offset_norm = jnp.sqrt(jnp.sum(offset**2, axis=-1) + 1e-5)
        offset_hat = offset / offset_norm[:, None]
        offset_norm_scale = jnp.tanh(offset_norm / tube_radius)

        return snap_anchors + tube_radius * offset_norm_scale[:, None] * offset_hat

    @property
    def exponents(self) -> Float[Array, "n_points"]:
        log_min, log_max = lax.stop_gradient(self.log_exponent_bounds)
        s = jax.nn.sigmoid(self.exponent_params)
        return jnp.exp(log_min + (log_max - log_min) * s)


@jax.jit
def _screened_coulomb(r, omega):
    eps = jnp.finfo(r.dtype).eps
    long_range = erf(omega * r) / r
    short_range = (2 * omega / jnp.sqrt(jnp.pi)) * (1 - ((omega * r) ** 2) / 3)
    return jnp.where(r > eps, long_range, short_range)


class MonopoleCloudPotential(eqx.Module):
    splat: AtomGaussianSplat
    charge_params: Float[Array, " num_monopoles"]
    excess_charge: float = eqx.field(static=True)

    def __init__(self, num_monopoles: int, *args, excess_charge: float = 0.0, **kwargs):
        self.splat = AtomGaussianSplat(num_monopoles, *args, **kwargs)
        self.charge_params = jnp.zeros(shape=(num_monopoles,))
        self.excess_charge = excess_charge

    @property
    def charges(self) -> Float[Array, " num_monopoles"]:
        q = self.charge_params - jnp.mean(self.charge_params)
        return q + self.excess_charge / self.splat.n_points

    def __call__(self, r: Float[Array, "3"]) -> Float[Array, " n_grid"]:
        d = jnp.linalg.norm(r - self.splat.positions, axis=-1)
        omega = jnp.sqrt(self.splat.exponents)
        return self.charges @ _screened_coulomb(d, omega)


class DipoleCloudPotential(eqx.Module):
    splat: AtomGaussianSplat
    moments: Float[Array, "num_dipoles 3"]

    def __init__(self, num_dipoles: int, *args, **kwargs):
        self.splat = AtomGaussianSplat(num_dipoles, *args, **kwargs)
        self.moments = jnp.zeros(shape=(num_dipoles, 3))

    def __call__(self, r: Float[Array, "3"]) -> Scalar:

        omega = jnp.sqrt(self.splat.exponents)

        def aux(R):
            d = jnp.linalg.norm(r - R, axis=-1)
            return _screened_coulomb(d, omega).sum()

        _, v_val = jax.jvp(aux, (self.splat.positions,), (self.moments,))
        return v_val


class MultipolePotential(eqx.Module):
    """Variational local potential represented by Gaussian monopole and dipole splats.

    Args:
        num_monopoles: Number of trainable Gaussian monopoles.
        num_dipoles: Number of trainable Gaussian dipoles.
        atom_coords: Nuclear coordinates in Bohr, with shape ``(n_atoms, 3)``.
        atom_charges: Nuclear charges with shape ``(n_atoms,)``.
        geometry_init: Initializer for splat positions and exponents.
        excess_charge: Required sum of all monopole charges. For a neutral-system OEP
            with exact-exchange fraction ``gamma``, use ``1 - gamma``.
        exponent_bounds: Optional positive lower and upper bounds for Gaussian
            exponents in inverse Bohr squared.
        tube_scale: Scale of the smooth molecular-neighbourhood position constraint.
        snap_scale: Scale controlling smooth assignment of splats to nearby atoms.
        key: JAX PRNG key used by ``geometry_init``.

    Calling the instance at ``r`` evaluates only the variational correction
    ``v_MS(r)``. Nuclear and Fermi--Amaldi background terms are not included. Monopole
    charges satisfy ``sum(q) == excess_charge`` for every parameter value; positions
    and exponents are constrained through smooth views of unconstrained parameters.
    """

    monopole: MonopoleCloudPotential
    dipole: DipoleCloudPotential

    def __init__(
        self, num_monopoles: int, num_dipoles: int, *args, excess_charge: float = 0.0, **kwargs
    ):
        key1, key2 = jr.split(kwargs.pop("key"), 2)
        self.monopole = MonopoleCloudPotential(
            num_monopoles, *args, excess_charge=excess_charge, **kwargs, key=key1
        )
        self.dipole = DipoleCloudPotential(num_dipoles, *args, **kwargs, key=key2)

    def __call__(self, r: Float[Array, "3"]) -> Scalar:
        return self.monopole(r) + self.dipole(r)


def _monopole_density(r, alpha):
    norm = jnp.sqrt(alpha / jnp.pi) ** (3 / 2)
    logit = -alpha * jnp.sum(r**2, axis=-1)
    return norm * jnp.exp(logit)


class MonopoleDensity(eqx.Module):
    potential: MonopoleCloudPotential

    def __call__(self, r: Float[Array, "3"]) -> Scalar:
        charges = self.potential.charges
        positions = self.potential.splat.positions
        exponents = self.potential.splat.exponents
        return charges @ _monopole_density(r - positions, exponents)


class DipoleDensity(eqx.Module):
    potential: DipoleCloudPotential

    def __call__(self, r: Float[Array, "3"]) -> Scalar:

        moments = self.potential.moments
        positions = self.potential.splat.positions
        exponents = self.potential.splat.exponents

        _, rho_val = jax.jvp(
            lambda R: _monopole_density(r - R, exponents).sum(), (positions,), (moments,)
        )

        return rho_val


class MultipoleDensity(eqx.Module):
    monopole: MonopoleDensity
    dipole: DipoleDensity

    def __init__(self, potential: MultipolePotential):
        self.monopole = MonopoleDensity(potential.monopole)
        self.dipole = DipoleDensity(potential.dipole)

    def __call__(self, r: Float[Array, "3"]) -> Scalar:
        return self.monopole(r) + self.dipole(r)


####################################################################################################
################# 3-center integrals for analytical multipole potential AO matrices ################
####################################################################################################


def _make_aux_mol(mol, positions, exponents, l):
    atom = [[f"X{i}", pos] for i, pos in enumerate(positions)]
    basis = {f"X{i}": [[l, (a, 1.0)]] for i, a in enumerate(exponents)}
    return gto.Mole(atom=atom, basis=basis, cart=mol.cart, unit="Bohr", verbose=0).build()


def _multipole_ao_mat(mol, aux_mol, with_pos_grad=False):

    nao = mol.nao_nr()
    npts = aux_mol.natm

    int_ = df.incore.aux_e2(mol, aux_mol, intor="int3c2e")
    return int_.reshape(nao, nao, npts, -1)

    # int_ = int_.reshape(nao, nao, npts, -1)

    # if not with_pos_grad:
    #     return int_

    # else:
    #     grad_int_ = df.incore.aux_e2(mol, aux_mol, intor="int3c2e_ip2")
    #     grad_int_ = grad_int_.reshape(3, *int_.shape)
    #     return int_, grad_int_


### Monopoles:


def _monopole_potential_mat_fwd_cpu(mol, charges, positions, exponents):

    aux_mol = _make_aux_mol(mol, positions, exponents, l=0)
    int_ = _multipole_ao_mat(mol, aux_mol, with_pos_grad=False).squeeze(-1)
    mat = np.dot(int_, charges)

    return mat  # , aux_mol


def _monopole_potential_mat_bwd_cpu(mol, cotangents, charges, positions, exponents):

    aux_mol = _make_aux_mol(mol, positions, exponents, l=0)
    ints, grad_ints = _multipole_ao_mat(mol, aux_mol, with_pos_grad=True)
    ints, grad_ints = ints.squeeze(-1), grad_ints.squeeze(-1)

    # ints.shape = (nao, nao, n_monopoles, 1)
    # grad_ints.shape = (3, nao, nao, n_monopoles, 1)

    grad_pos = -np.einsum("mn,imnk,k->ki", cotangents, grad_ints, charges)
    grad_charges = np.einsum("mn,mnk->k", cotangents, ints)

    # Exponent grads:
    aux_mol_d = _make_aux_mol(mol, positions, exponents, l=2)
    d_mat = _multipole_ao_mat(mol, aux_mol_d, with_pos_grad=False)

    grad_shape_part = -(d_mat[..., 0] + d_mat[..., 3] + d_mat[..., 5])
    grad_norm_part = (3 / (4 * exponents)) * ints  # Broadcast: (N, N, K) * (K,)
    grad_exp_ = grad_shape_part + grad_norm_part

    grad_exp = np.einsum("mn,k,mnk->k", cotangents, charges, grad_exp_)

    return grad_charges, grad_pos, grad_exp


def _dipole_potential_mat_fwd_cpu(mol, moments, positions, exponents):

    aux_mol = _make_aux_mol(mol, positions, exponents, l=1)
    int_ = _multipole_ao_mat(mol, aux_mol, with_pos_grad=False)
    mat = np.tensordot(int_, moments, axes=2)

    return mat  # , aux_mol


def _dipole_potential_mat_bwd_cpu(mol, cotangents, moments, positions, exponents):

    aux_mol = _make_aux_mol(mol, positions, exponents, l=1)
    ints, grad_ints = _multipole_ao_mat(mol, aux_mol, with_pos_grad=True)

    # ints.shape = (nao, nao, n_dipoles, 3)
    # grad_ints.shape = (3, nao, nao, n_dipoles, 3)

    # grad_{position} = -grad_{center}
    grad_pos = -np.einsum("mn,imnkc,kc->ki", cotangents, grad_ints, moments)
    grad_moments = np.einsum("mn,mnkc->kc", cotangents, ints)

    # Exponent grads:
    aux_mol_f = _make_aux_mol(mol, positions, exponents, l=3)
    f_mat = _multipole_ao_mat(mol, aux_mol_f, with_pos_grad=False)

    dx_exp = -(f_mat[..., 0] + f_mat[..., 3] + f_mat[..., 5])
    dy_exp = -(f_mat[..., 1] + f_mat[..., 6] + f_mat[..., 8])
    dz_exp = -(f_mat[..., 2] + f_mat[..., 7] + f_mat[..., 9])

    grad_shape_part = np.stack([dx_exp, dy_exp, dz_exp], axis=-1)  # shape = (N, N, K, 3)
    grad_norm_part = (5 / (4 * exponents[..., None])) * ints  # Broadcast: (K, 1) * (N, N, K, 3)
    grad_exp_ = grad_shape_part + grad_norm_part

    grad_exp = np.einsum("mn,kc,mnkc->k", cotangents, moments, grad_exp_)

    return grad_moments, grad_pos, grad_exp


def _wrap_fwd(fn):
    def wrapped(mol, *params):
        nao = mol.nao_nr()
        mat_shape = jax.ShapeDtypeStruct((nao, nao), params[0].dtype)
        return eqx.filter_pure_callback(
            fn, mol, *params, result_shape_dtypes=mat_shape, vmap_method="broadcast_all"
        )

    return wrapped


def _wrap_bwd(fn):
    def wrapped(mol, cotangents, *params):
        out_shape = eqx.filter_eval_shape(lambda: params)
        return eqx.filter_pure_callback(
            fn, mol, cotangents, *params, result_shape_dtypes=out_shape, vmap_method="broadcast_all"
        )

    return wrapped


_monopole_potential_mat_fwd_wrapped = _wrap_fwd(_monopole_potential_mat_fwd_cpu)
_monopole_potential_mat_bwd_wrapped = _wrap_bwd(_monopole_potential_mat_bwd_cpu)
_dipole_potential_mat_fwd_wrapped = _wrap_fwd(_dipole_potential_mat_fwd_cpu)
_dipole_potential_mat_bwd_wrapped = _wrap_bwd(_dipole_potential_mat_bwd_cpu)


@eqx.filter_custom_vjp
def _monopole_potential_mat(params, mol):
    return _monopole_potential_mat_fwd_wrapped(mol, *params)


@_monopole_potential_mat.def_fwd
def _monopole_potential_mat_fwd(_, params, mol):
    mat = _monopole_potential_mat_fwd_wrapped(mol, *params)
    return mat, None


@_monopole_potential_mat.def_bwd
def _monopole_potential_mat_bwd(_, d_mat, __, params, mol):
    return _monopole_potential_mat_bwd_wrapped(mol, d_mat, *params)


@eqx.filter_custom_vjp
def _dipole_potential_mat(params, mol):
    return _dipole_potential_mat_fwd_wrapped(mol, *params)


@_dipole_potential_mat.def_fwd
def _dipole_potential_mat_fwd(_, params, mol):
    mat = _dipole_potential_mat_fwd_wrapped(mol, *params)
    return mat, None


@_dipole_potential_mat.def_bwd
def _dipole_potential_mat_bwd(_, d_mat, __, params, mol):
    return _dipole_potential_mat_bwd_wrapped(mol, d_mat, *params)


def monopole_potential_matrix(
    mol: gto.Mole,
    charges: Float[Array, " n_monopoles"],
    positions: Float[Array, " n_monopoles 3"],
    exponents: Float[Array, " n_monopoles"],
) -> Float[Array, "nao nao"]:
    return _monopole_potential_mat((charges, positions, exponents), mol)


def dipole_potential_matrix(
    mol: gto.Mole,
    moments: Float[Array, "n_dipoles 3"],
    positions: Float[Array, " n_dipoles 3"],
    exponents: Float[Array, " n_dipoles"],
) -> Float[Array, "nao nao"]:
    return _dipole_potential_mat((moments, positions, exponents), mol)


class MultipolePotentialMatrix(eqx.Module):
    """Project a multipole-splat potential analytically into a molecular AO basis.

    Args:
        mol: PySCF molecule defining the target AO basis. Auxiliary centers and
            exponents are interpreted in atomic units.
        chunk_size: Number of splats processed per callback. Use ``None`` to project
            all splats at once.

    Calling the instance returns the matrix with elements
    ``integral chi_m(r) v_MS(r) chi_n(r) dr``. Three-center Coulomb integrals are
    evaluated in a Cartesian auxiliary representation and transformed back when the
    molecular AO basis is spherical.
    """

    _mol_cart: gto.Mole = eqx.field(static=True)
    _c2s_coeff: Float[Array, "nao_cart nao"] | None
    _chunk_size: int | None
    nao: int

    def __init__(self, mol: gto.Mole, chunk_size: int | None = 256):

        if mol.cart:
            self._mol_cart = mol.build()
            self._c2s_coeff = None

        else:
            mol_cart = mol.copy()
            mol_cart.cart = True

            self._mol_cart = mol_cart.build()
            self._c2s_coeff = jnp.array(mol.cart2sph_coeff(normalized="sp"))

        self.nao = self._mol_cart.nao_nr()
        self._chunk_size = chunk_size

    def __call__(self, potential: MultipolePotential) -> Float[Array, "nao nao"]:

        mat = self._monopole(potential.monopole) + self._dipole(potential.dipole)

        if self._c2s_coeff is not None:
            c2s = self._c2s_coeff  # Cartesian to Spherical basis coefficients
            mat = jnp.einsum("pm,qn,pq->mn", c2s, c2s, mat)

        return mat

    def _monopole(self, potential):

        init_value = jnp.zeros((self.nao, self.nao), dtype=potential.splat.exponents.dtype)

        norm_corr = (potential.splat.exponents / (2 * jnp.pi)) ** (3 / 4)
        rescaled_charges = potential.charges * norm_corr

        return reduce(
            lambda *p: monopole_potential_matrix(self._mol_cart, *p),
            rescaled_charges,
            potential.splat.positions,
            potential.splat.exponents,
            init_value=init_value,
            chunk_size=self._chunk_size,
        )

    def _dipole(self, potential):

        init_value = jnp.zeros((self.nao, self.nao), dtype=potential.splat.exponents.dtype)

        norm_corr = (potential.splat.exponents ** (5 / 4)) / ((2 * jnp.pi) ** (3 / 4))
        rescaled_moments = potential.moments * norm_corr[:, None]

        return reduce(
            lambda *p: dipole_potential_matrix(self._mol_cart, *p),
            rescaled_moments,
            potential.splat.positions,
            potential.splat.exponents,
            init_value=init_value,
            chunk_size=self._chunk_size,
        )
