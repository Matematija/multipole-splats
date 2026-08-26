from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, Key, PyTree
from pyscf import dft, gto

from .utils import reduce


class Grid(eqx.Module):
    coords: Float[Array, "n_grid 3"] = eqx.field(converter=jnp.asarray)
    weights: Float[Array, " n_grid"] = eqx.field(converter=jnp.asarray)

    def __len__(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def from_pyscf(
        cls,
        mol: gto.Mole,
        level: int = 2,
        radi_method: Callable = dft.radi.treutler_ahlrichs,
        prune: Callable | None = None,
    ) -> "Grid":

        grids = dft.gen_grid.Grids(mol)
        grids.radi_method = radi_method
        grids.level = level
        grids.prune = prune
        grids.build()

        return cls(grids.coords, grids.weights)

    def subsample(self, n_samples: int, key: Key) -> tuple["Grid", Int[Array, " n_samples"]]:

        size = len(self)
        assert n_samples < size, "Number of points to sample must be less than the grid size."

        indices = jax.random.choice(key, size, shape=(n_samples,), replace=False)
        coords_ = self.coords[indices]
        weights_ = (size / n_samples) * self.weights[indices]

        return Grid(coords_, weights_), indices


def grid_integrate(
    fn: Callable[[Float[Array, "3"]], PyTree], grid: Grid, chunk_size: int | None = None
) -> PyTree:
    """Compute sum_x weights[x] * fn(coords[x]) over the grid.

    fn receives a single point r: (3,) and returns a PyTree contribution.
    With chunk_size, processes the grid in memory-bounded chunks via lax.scan.
    """

    vmap_fn = jax.vmap(fn)

    @partial(jax.checkpoint, prevent_cse=False)  # pyright: ignore
    def weighted_sum(coords_chunk, weights_chunk):
        results = vmap_fn(coords_chunk)
        return jax.tree.map(lambda r: jnp.tensordot(weights_chunk, r, axes=1), results)

    if chunk_size is None:
        return weighted_sum(grid.coords, grid.weights)

    abstract = jax.eval_shape(weighted_sum, grid.coords[:chunk_size], grid.weights[:chunk_size])
    init_val = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), abstract)

    def reduce_fn(a, b):
        return jax.tree.map(jnp.add, a, b)

    return reduce(
        weighted_sum,
        grid.coords,
        grid.weights,
        reduce_fn=reduce_fn,
        init_value=init_val,
        chunk_size=chunk_size,
    )
