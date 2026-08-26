from collections.abc import Callable, Sequence
from functools import partial, wraps
from typing import Any

import equinox as eqx
import jax
from jax import Array, lax
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float, PyTree, Scalar, ScalarLike

Shape = Sequence[int]
DType = DTypeLike


def curry(f: Callable) -> Callable:
    return partial(partial, f)


def tree_size(tree: PyTree) -> int:
    return sum(jnp.size(l) for l in jax.tree.leaves(tree))


def tree_shape(tree: PyTree) -> PyTree:
    return jax.tree.map(jnp.shape, tree)


def tree_dot(left: PyTree, right: PyTree, conj: bool = False) -> Scalar:

    if conj:
        left = jax.tree.map(jnp.conjugate, left)

    aux = jax.tree.map(lambda a, b: jnp.tensordot(a, b, axes=a.ndim), left, right)
    return jax.tree.reduce(jnp.add, aux)


def abs2(z: Array) -> Array:
    if jnp.iscomplexobj(z):
        return jnp.real(z) ** 2 + jnp.imag(z) ** 2
    else:
        return z**2


def default_dtype() -> DType:
    return jnp.float64 if jax.config.x64_enabled else jnp.float32  # pyright: ignore


def default_int_dtype() -> DType:
    return jnp.int64 if jax.config.x64_enabled else jnp.int32  # pyright: ignore


@eqx.filter_custom_jvp
def _eigh(matrix, _, *, lower, eigvals_only):
    return jax.scipy.linalg.eigh(matrix, lower=lower, eigvals_only=eigvals_only)


@_eigh.def_jvp
def _eigh_jvp(primals, tangents, *, lower, eigvals_only):

    matrix, eps = primals
    d_matrix, d_eps = tangents
    assert d_eps is None, "Derivative with respect to eps is not supported."

    eigvals, eigvecs = _eigh(matrix, eps, lower=lower, eigvals_only=False)

    if eigvals_only:
        d_eigvals = jnp.einsum("mi,mn,ni->i", eigvecs.conj(), d_matrix, eigvecs)
        return eigvals, d_eigvals

    dM = jnp.einsum("mi,mn,nj->ij", eigvecs.conj(), d_matrix, eigvecs)

    diff = eigvals[:, None] - eigvals[None, :]
    inv_diff = jnp.where(jnp.abs(diff) > eps, 1 / diff, 0)

    d_eigvecs = jnp.matmul(eigvecs, -inv_diff * dM)  # pyright: ignore
    d_eigvals = jnp.diag(dM)

    return (eigvals, eigvecs), (d_eigvals, d_eigvecs)


@wraps(jax.scipy.linalg.eigh)
def eigh(
    matrix: Float[Array, "m m"],
    eps: ScalarLike | None = None,
    lower: bool = True,
    eigvals_only: bool = False,
) -> tuple[Float[Array, " m"], Float[Array, "m m"]] | Float[Array, " m"]:

    if eps is None:
        eps = jnp.finfo(matrix.dtype).eps

    return _eigh(matrix, eps, lower=lower, eigvals_only=eigvals_only)


@jax.jit
def psd_inv_sqrt(
    matrix: Float[Array, "... m n"], rcond: ScalarLike = 1e-5, acond: ScalarLike = 0.0
) -> Float[Array, "... m n"]:

    s, U = eigh(matrix)

    eps = jnp.finfo(matrix.dtype).eps
    acond = jnp.maximum(acond, eps)
    cutoff = jnp.maximum(acond, rcond * lax.stop_gradient(s.max()))

    inv_sqrt_s = jnp.where(s > cutoff, 1 / jnp.sqrt(s), 0.0)

    return jnp.einsum("...i,...mi,...ni->...mn", inv_sqrt_s, U, U)


####################################################################################################


def _argnums_partial(fun, args, dyn_argnums):

    sentinel = object()
    args_template = [sentinel] * len(args)
    dyn_args = []

    for i, arg in enumerate(args):
        if i in dyn_argnums:
            dyn_args.append(arg)
        else:
            args_template[i] = arg

    def fun_partial(*new_dyn_args):

        arg_iter = iter(new_dyn_args)

        interpolated_args = tuple(
            next(arg_iter) if arg == sentinel else arg for arg in args_template
        )

        return fun(*interpolated_args)

    return fun_partial, dyn_args


def _transpose_vmap_output(y, oax):
    if oax is None or oax == 0:
        return y
    else:
        return jnp.moveaxis(y, 0, oax)


def _transpose_vmap_outputs(outputs, axes):  # What a mess this is

    if len(axes) == 1:
        axes, outputs = (axes,), (outputs,)
        unpack = True
    else:
        unpack = False

    assert len(outputs) == len(axes)

    out = tuple(
        jax.tree.map(lambda l: _transpose_vmap_output(l, oax), leaf)  # noqa: B023
        for leaf, oax in zip(outputs, axes)
    )

    return out[0] if unpack else out


def _to_shape(x: int | Shape) -> Shape:
    return (x,) if isinstance(x, int) else x


def vmap(
    fun: Callable,
    in_axes: int | Shape = 0,
    out_axes: int | Shape = 0,
    chunk_size: int | None = None,
    *args,
    **kwargs,
) -> Callable:

    if chunk_size is None:
        return jax.vmap(fun, in_axes, out_axes, *args, **kwargs)

    in_axes = _to_shape(in_axes)
    argnums = tuple(i for i, ix in enumerate(in_axes) if ix is not None)

    if not set(in_axes).issubset((0, None)):
        _in_axes = [ix % len(in_axes) for ix in in_axes if ix is not None]

        def preprocess_dyn_args(dyn_args):
            return jax.tree.map(jnp.moveaxis, dyn_args, _in_axes, [0] * len(_in_axes))
    else:
        preprocess_dyn_args = lambda x: x  # pyright: ignore

    if not set(_to_shape(out_axes)).issubset((0, None)):
        postprocess_output = _transpose_vmap_outputs
    else:
        postprocess_output = lambda x, *_: x

    def f_chunked(*args, **kwargs):
        f_partial, dyn_args = _argnums_partial(partial(fun, **kwargs), args, argnums)
        dyn_args = preprocess_dyn_args(dyn_args)
        out = lax.map(lambda args: f_partial(*args), dyn_args, batch_size=chunk_size)
        return postprocess_output(out, _to_shape(out_axes))

    return f_chunked


class _VmapWrapper(eqx.Module):
    _fun: Callable
    _in_axes: int | Shape
    _out_axes: int | Shape
    _chunk_size: int | None

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, *args, **kwargs):

        dynamic_args, static_args = eqx.partition(args, eqx.is_inexact_array)

        @partial(vmap, in_axes=self._in_axes, out_axes=self._out_axes, chunk_size=self._chunk_size)
        def vmap_aux(*dyn_args):
            args = eqx.combine(dyn_args, static_args)
            return self._fun(*args, **kwargs)

        return vmap_aux(*dynamic_args)


def filter_vmap(
    fun: Callable,
    in_axes: int | Shape = 0,
    out_axes: int | Shape = 0,
    chunk_size: int | None = None,
) -> Callable:

    return _VmapWrapper(fun, in_axes, out_axes, chunk_size)


####################################################################################


def reduce(
    fn: Callable,
    *args: PyTree,
    reduce_fn: Callable[[Array, Array], Array] = jnp.add,
    init_value: Any = None,
    chunk_size: int | None = None,
) -> PyTree:

    if chunk_size is None:
        return fn(*args)

    total_size = jax.tree.leaves(args)[0].shape[0]
    n_chunks, remainder = divmod(total_size, chunk_size)
    bulk_size = n_chunks * chunk_size

    args_ = jax.tree.map(
        lambda l: jnp.reshape(l[:bulk_size], (n_chunks, chunk_size) + l.shape[1:]), args
    )

    def scan_fn(carry, xs):
        carry_ = reduce_fn(carry, fn(*xs))
        return carry_, None

    result, _ = lax.scan(scan_fn, init_value, args_)

    if remainder > 0:
        last_args = jax.tree.map(lambda l: l[-remainder:], args)
        last_result = fn(*last_args)
        result = reduce_fn(result, last_result)

    return result
