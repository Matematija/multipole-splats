# pyright: reportIncompatibleVariableOverride=false

from typing import ClassVar

import equinox as eqx
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Scalar

from ..basis import AtomicOrbitals, Integrals
from ..basis.energy import exchange_energy
from ..grid import Grid
from .base import LocalFunctional, XCFunctional
from .libxc import LibXCEnergyDensity


class ExactExchange(XCFunctional):
    name: ClassVar[str] = "EXX"
    xc_type: ClassVar[str] = "EXX"
    exx_fraction: ClassVar[float] = 1.0
    cderi: Float[Array, "naux nao nao"] = eqx.field(converter=jnp.asarray)

    def __call__(self, dm: Float[Array, "*spin nao nao"]) -> Scalar:
        return exchange_energy(self.cderi, dm)


class HybridFunctional(XCFunctional):
    exx: ExactExchange
    local: LocalFunctional
    exx_fraction: eqx.AbstractVar[float]

    def __call__(self, dm: Float[Array, "*spin nao nao"]) -> Scalar:
        return self.exx_fraction * self.exx(dm) + self.local(dm)


class PBE0(HybridFunctional):
    name: ClassVar[str] = "PBE0"
    xc_type: ClassVar[str] = "GGA"
    exx_fraction: ClassVar[float] = 0.25

    def __init__(self, ints: Integrals, ao: AtomicOrbitals, grid: Grid, *, spin: bool = True):
        local_part = LibXCEnergyDensity("PBE0", spin=spin)  # LibXC skips the EXX part
        self.local = LocalFunctional(local_part, ao, grid)
        self.exx = ExactExchange(ints.cderi)


class B3LYP(HybridFunctional):
    name: ClassVar[str] = "B3LYP"
    xc_type: ClassVar[str] = "GGA"
    exx_fraction: ClassVar[float] = 0.2

    def __init__(self, ints: Integrals, ao: AtomicOrbitals, grid: Grid, *, spin: bool = True):
        local_part = LibXCEnergyDensity("B3LYP", spin=spin)  # LibXC skips the EXX part
        self.local = LocalFunctional(local_part, ao, grid)
        self.exx = ExactExchange(ints.cderi)
