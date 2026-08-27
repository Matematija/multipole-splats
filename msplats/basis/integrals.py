from typing import Any

import equinox as eqx
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Scalar
from pyscf import df, gto


class Integrals(eqx.Module):
    """Molecular AO integrals used by the differentiable energy expressions.

    Args:
        mol: PySCF molecule. It is built in place before the integrals are evaluated.
        auxbasis: Optional PySCF auxiliary basis specification. When omitted, PySCF
            chooses a density-fitting auxiliary basis.

    The stored arrays are the overlap, kinetic and nuclear-attraction matrices, a
    three-index density-fitting factor for the electron-repulsion integrals, and the
    nuclear-repulsion energy. Energies and coordinates follow PySCF atomic units.
    Recreate this object whenever the molecular geometry or AO basis changes.
    """

    ovlp: Float[Array, "nao nao"]
    int1e_kin: Float[Array, "nao nao"]
    int1e_nuc: Float[Array, "nao nao"]
    cderi: Float[Array, "naux nao nao"]
    energy_nuc: Scalar

    def __init__(self, mol: gto.Mole, auxbasis: Any = None):

        mol.build()

        self.energy_nuc = jnp.asarray(mol.energy_nuc())
        self.ovlp = jnp.asarray(mol.intor("int1e_ovlp"))
        self.int1e_kin = jnp.asarray(mol.intor("int1e_kin"))
        self.int1e_nuc = jnp.asarray(mol.intor("int1e_nuc"))

        # self.eri = jnp.asarray(mol.intor("int2e", aosym="s8"))

        if auxbasis is None:
            auxbasis = df.make_auxbasis(mol)

        cderi_flat = df.incore.cholesky_eri(mol, auxbasis, aosym="s1")
        self.cderi = jnp.asarray(cderi_flat.reshape(-1, mol.nao, mol.nao))

    @property
    def int_1e(self) -> Array:
        return self.int1e_kin + self.int1e_nuc
