# AGENTS.md

## Purpose

This file applies to the entire repository. `multipole-splats` is a compact research library for
inverse Kohn--Sham (IKS) and optimized effective potential (OEP) calculations using JAX, Equinox,
and PySCF. Preserve its small, mathematical character. Prefer direct scientific code over framework
machinery, and do not add speculative generality or restore abandoned experiments without a concrete
need.

Before editing, inspect the worktree and read the complete modules involved. Preserve user changes,
including untracked files, and avoid unrelated cleanup.

## Scientific model

The code optimizes a local one-electron potential through the differentiable path

`potential parameters -> AO potential matrix -> generalized eigensystem -> orbitals/density -> loss`.

The trial potential separates a fixed nuclear and Fermi--Amaldi (FA) background from a variational
multipole-splat contribution,

```text
v_theta(r) = v_ext(r) + v_FA(r) + v_MS(r)
v_FA(r) = (N - 1) / N * integral n_ref(r') / |r - r'| dr'
v_MS(r) = sum_k q_k v_k(r) - sum_k p_k . grad v_k(r)
v_k(r) = erf(sqrt(alpha_k) |r - a_k|) / |r - a_k|
```

`MultipolePotential` represents `v_MS`, not necessarily the complete `v_theta`. Fixed nuclear and FA
terms may already be stored in the one-electron Hamiltonian assembled by the eigensolver or loss.
Trace that assembly before changing it; adding a background twice is a physically different model.

Each `v_k` is sourced by the normalized Gaussian

```text
rho_k(r) = (alpha_k / pi)^(3/2) exp(-alpha_k |r - a_k|^2),
```

so `laplacian(v_k) = -4 pi rho_k`. Finite widths keep both sources and potentials analytic at their
centers. The dipole sign follows differentiation with respect to the spatial coordinate; take care
when implementing it through derivatives with respect to the Gaussian center.

The monopole charge constraint is part of the parameterization:

```text
q_k = w_k - mean(w) + excess_charge / n_monopoles
sum_k q_k = excess_charge.
```

For a neutral system with exact-exchange fraction `gamma`, use
`excess_charge = 1 - gamma`. Together with the nuclear and FA backgrounds, this fixes the total
long-range tail to `-gamma / r` for every parameter value. Do not replace this identity with a penalty
or post-update correction. Positions are smoothly confined to the molecular neighbourhood and
exponents to positive bounds; optimize unconstrained raw parameters and expose physical values through
derived properties.

The generalized eigenproblem is `H C = S C epsilon`. Restricted density matrices have shape
`(nao, nao)` and are spin-summed; unrestricted matrices have shape `(2, nao, nao)` with spin first.
Occupation and spin factors in the density, Hartree, and exchange expressions depend on this convention.
Derive them explicitly rather than inferring them from broadcasting. Reference densities and density
matrices must use the same geometry, AO basis, spin convention, and occupations as the trial system.

The two optimization objectives share the same potential-to-density map:

- OEP minimizes the orbital-dependent generalized Kohn--Sham energy over densities induced by a local
  potential.
- IKS maximizes the Wu--Yang functional
  `W[v] = T_s[phi[v]] + integral v(r) (n[v](r) - n_ref(r)) dr`, or equivalently minimizes `-W`.
  Its potential gradient is the density residual `n[v] - n_ref`; no KS-response inversion is needed.
- Force-matching regularization has the schematic form
  `R[v] proportional to integral sigma(r) |grad v(r) - grad v_0(r)|^2 dr`, where `sigma` is one or a
  fixed reference-density weight. When the fixed background is separated analytically,
  the optimized term reduces to the field of `v_MS`. Regularization selects a smoother representative
  among nearly degenerate potentials; it is not required for convergence and does not remove the
  null space of the unregularized KS response.

The analytic multipole density is a potential *source density*, not an XC hole. When used for the XC
source, its smoothness and total charge are guaranteed by the ansatz; neither is an empirical result
or convergence test.

## Code map

- `msplats/basis/`: differentiable AO evaluation, PySCF integrals, orbital/density objects,
  eigensolvers, and energy terms.
- `msplats/potential/multipole.py`: splat geometry and constraints, real-space sources/potentials, and
  analytic three-center AO matrix elements.
- `msplats/xc/`: differentiable LibXC-backed local, exact-exchange, and hybrid functionals.
- `msplats/grid.py`: molecular quadrature and memory-bounded integration.
- `msplats/loss.py`: OEP/IKS objectives and regularization.
- `msplats/data.py`: atomic radii and other fixed element data, stored in atomic units.
- `msplats/utils.py`: small JAX/PyTree, eigensolver, vectorization, and chunking primitives.

Keep these boundaries narrow. Reuse utilities instead of creating local variants, and keep
`__init__.py` files limited to intentional public exports.

A typical calculation starts from one PySCF `Mole`. `Integrals` builds overlap, one-electron, and
Cholesky-factorized two-electron data; `AtomicOrbitals` and `Grid` provide real-space evaluation;
an XC functional and `EnergyFunctional` define the target energy; `MultipolePotential` defines the
trainable real-space correction; `MultipolePotentialMatrix` projects it analytically into the AO
basis; and `SingleElectronProblem` maps that matrix to orbitals. The loss closes the loop by constructing
the density matrix and evaluating the OEP or IKS objective. Integrals, AO/grid caches, and PySCF objects
are geometry- and basis-specific: rebuild them when either changes.

## Coding style

Match surrounding code, including conventions not enforced by Ruff:

- Target Python 3.10 or newer. Project metadata, basedpyright, and Ruff all use 3.10 as the language
  floor; do not introduce 3.11+ syntax without updating them together. Use double quotes,
  100-character lines, and Ruff import order. Import runtime protocols from `collections.abc`; use
  modern unions and built-in generics.
- Use `jax.numpy` as `jnp` on traced paths and NumPy only at explicit host/PySCF boundaries.
- Model stateful callable scientific objects as `eqx.Module` classes. Declare fields together at the
  class top, mark only genuinely static JAX data with `eqx.field(static=True)`, and use array converters
  at input boundaries where appropriate.
- Add jaxtyping annotations to public numerical interfaces and Equinox fields. Keep shape names such as
  `nao`, `naux`, `n_grid`, and the leading `*spin` convention consistent across modules.
- Keep PRNG flow explicit. Accept a keyword-only `key` for random construction and split keys near use.
- Prefer compact equations using `einsum`, `tensordot`, broadcasting, and JAX transforms over expanded
  loops. Short mathematical locals (`dm`, `ao`, `E1`, `Exc`, `C`) are idiomatic; public names remain
  descriptive.
- Prefix internal helpers and implementation-only state with `_`. Use properties for inexpensive
  derived quantities and constrained views of trainable parameters.
- Preserve the vertical rhythm: a blank line before nontrivial function bodies and between mathematical
  steps. Avoid deep nesting.
- Comments and docstrings should explain physical meaning, shapes, numerical stability, or non-obvious
  derivatives, not narrate obvious code. Keep type-checker suppressions local.
- Raise clear errors for unsupported shapes or theory branches. Do not silently coerce a scientifically
  different case.

Prefer a direct implementation until an abstraction has multiple real call sites. Remove dead code
instead of retaining large commented alternatives.

## JAX and PySCF boundaries

PySCF objects and kernels are not traceable. Route them through `jax.pure_callback` or
`eqx.filter_pure_callback` with explicit output shape and dtype contracts. A callback used in
differentiation needs a validated custom JVP/VJP; autodiff cannot pass through host code.

Inside transformed code:

- avoid data-dependent Python control flow, host conversions, mutation, and hidden global state;
- preserve dtype and state whether scientific comparisons use JAX x64;
- use `lax.stop_gradient` for intentionally fixed caches and reference data;
- use the existing chunked `vmap` and `reduce` helpers and keep chunked and unchunked results equal;
- choose callback `vmap_method` deliberately and test the transformations actually used.

Preserve small-distance analytic limits, PSD/overlap cutoffs, eigenvalue-degeneracy handling,
Cartesian-to-spherical transformations, and Gaussian normalization factors unless a replacement is
scientifically justified and numerically tested. Internal molecular coordinates and auxiliary centers
are in Bohr; energies are in Hartree unless an interface explicitly states otherwise.

## Validation and change discipline

Standard checks from the repository root are:

```bash
python -m pip install -e ".[dev]"
ruff format --check msplats
ruff check msplats
basedpyright
```

There is not yet a committed automated test suite, so linting is not scientific validation. For a
numerical change, run the smallest relevant comparison and report it explicitly: values against a
PySCF/NumPy reference, custom derivatives against finite differences or an independent contraction,
and relevant restricted/unrestricted, Cartesian/spherical, or chunked/unchunked cases. Exercise edge
conditions such as coincident centers, small eigenvalue gaps, and exponent bounds. Use small molecules
and dtype-aware tolerances; separate pre-existing failures from regressions.

Keep patches focused and preserve public behavior unless the task changes it. Any change to an equation,
contraction, normalization, sign, spin factor, unit, asymptotic constraint, or custom derivative needs
a scientific explanation and a numerical check. Update exports and nearby shape annotations together
with the implementation.
