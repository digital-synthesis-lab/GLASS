"""Network — positions + cell + topology (bonds, adjacency)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


MAX_DEGREE = 4


@dataclass
class Network:
    """4-coordinated periodic network of Si atoms.

    `bonds` is the canonical store; `neigh` is a redundant adjacency view
    rebuilt from `bonds` for O(1) lookups during proposals and BFS.

    Atoms in `neigh[i]` are padded with -1 when `degree[i] < 4` (this
    happens transiently during loop-expansion initialization).
    """

    positions: np.ndarray  # (N, 3) float64
    cell: np.ndarray  # (3, 3) float64, orthorhombic in v1
    bonds: np.ndarray  # (M, 2) int64, undirected, each pair stored once
    neigh: np.ndarray  # (N, 4) int64, -1 padded
    degree: np.ndarray  # (N,) int64
    d: float = 2.35  # Å — Keating equilibrium bond length (scales with density)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    @classmethod
    def from_bonds(
        cls,
        positions: np.ndarray,
        cell: np.ndarray,
        bonds: np.ndarray,
        d: float = 2.35,
    ) -> "Network":
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        cell = np.ascontiguousarray(cell, dtype=np.float64)
        bonds = np.ascontiguousarray(bonds, dtype=np.int64)
        n = positions.shape[0]
        neigh, degree = _build_neigh(bonds, n)
        return cls(positions, cell, bonds, neigh, degree, float(d))

    def rebuild_neigh(self) -> None:
        self.neigh, self.degree = _build_neigh(self.bonds, self.n_atoms)

    def has_bond(self, i: int, j: int) -> bool:
        return _has_neigh(self.neigh, self.degree, int(i), int(j))

    def add_bond(self, i: int, j: int) -> None:
        if i == j:
            raise ValueError("self-bond")
        if self.has_bond(i, j):
            return
        di, dj = int(self.degree[i]), int(self.degree[j])
        if di >= MAX_DEGREE or dj >= MAX_DEGREE:
            raise RuntimeError(f"degree overflow at bond {i}-{j}")
        self.neigh[i, di] = j
        self.neigh[j, dj] = i
        self.degree[i] = di + 1
        self.degree[j] = dj + 1
        self.bonds = np.vstack([self.bonds, np.array([[i, j]], dtype=np.int64)])

    def remove_bond(self, i: int, j: int) -> None:
        _remove_neigh_entry(self.neigh, self.degree, int(i), int(j))
        _remove_neigh_entry(self.neigh, self.degree, int(j), int(i))
        # Drop from bonds array.
        a, b = (i, j) if i < j else (j, i)
        b0 = np.minimum(self.bonds[:, 0], self.bonds[:, 1])
        b1 = np.maximum(self.bonds[:, 0], self.bonds[:, 1])
        mask = ~((b0 == a) & (b1 == b))
        self.bonds = self.bonds[mask].copy()

    def replace_bond(self, old_i: int, old_j: int, new_i: int, new_j: int) -> None:
        """Remove (old_i, old_j) and add (new_i, new_j) atomically."""
        self.remove_bond(old_i, old_j)
        self.add_bond(new_i, new_j)

    def copy(self) -> "Network":
        return Network(
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            bonds=self.bonds.copy(),
            neigh=self.neigh.copy(),
            degree=self.degree.copy(),
            d=self.d,
        )


def _build_neigh(bonds: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    neigh = np.full((n_atoms, MAX_DEGREE), -1, dtype=np.int64)
    degree = np.zeros(n_atoms, dtype=np.int64)
    for i, j in bonds:
        di, dj = degree[i], degree[j]
        if di >= MAX_DEGREE or dj >= MAX_DEGREE:
            raise RuntimeError(
                f"degree overflow building neigh at bond {i}-{j} "
                f"(degree[{i}]={di}, degree[{j}]={dj})"
            )
        neigh[i, di] = j
        neigh[j, dj] = i
        degree[i] = di + 1
        degree[j] = dj + 1
    return neigh, degree


def _has_neigh(neigh: np.ndarray, degree: np.ndarray, i: int, j: int) -> bool:
    for k in range(degree[i]):
        if neigh[i, k] == j:
            return True
    return False


def _remove_neigh_entry(
    neigh: np.ndarray, degree: np.ndarray, i: int, j: int
) -> None:
    di = int(degree[i])
    for k in range(di):
        if neigh[i, k] == j:
            neigh[i, k] = neigh[i, di - 1]
            neigh[i, di - 1] = -1
            degree[i] = di - 1
            return
    raise RuntimeError(f"bond {i}-{j} not present in neighbor list")


def cubic_cell_for_si(n_atoms: int, density_g_cm3: float = 2.33) -> np.ndarray:
    """Cubic orthorhombic cell sized for `n_atoms` Si atoms at given density."""
    # 28.0855 g/mol Si, 6.022e23 /mol, 1 Å³ = 1e-24 cm³
    mass_g = n_atoms * 28.0855 / 6.022_140_76e23
    volume_cm3 = mass_g / density_g_cm3
    volume_A3 = volume_cm3 * 1e24
    a = volume_A3 ** (1.0 / 3.0)
    return np.diag([a, a, a]).astype(np.float64)


# Reference: crystalline Si at d=2.35 Å. We anchor the ρ↔d map at
# diamond geometry exactly — V_atom = a³/8 with a = 4d/√3 — so that
# bond_length_for_density and cubic_cell_for_si are self-consistent.
_REF_BOND_LENGTH = 2.35  # Å — diamond Si bond length

# m_Si / N_A in g, used by both helpers.
_M_SI_G = 28.0855 / 6.022_140_76e23


def _density_from_bond_length(d: float) -> float:
    """ρ (g/cm³) of an ideal-diamond network with bond length d (Å)."""
    a = 4.0 * d / np.sqrt(3.0)
    v_atom_A3 = (a**3) / 8.0
    return _M_SI_G / (v_atom_A3 * 1e-24)


_REF_DENSITY = _density_from_bond_length(_REF_BOND_LENGTH)  # ≈ 2.3344 g/cm³


def bond_length_for_density(density_g_cm3: float = 2.33) -> float:
    """Equilibrium Keating bond length d at the given mass density.

    Uses isotropic scaling d = d_ref · (ρ_ref/ρ)^(1/3), anchored at
    crystalline Si geometry. The bond list defines topology; this
    controls how the Keating energy is centred under that topology.
    """
    if density_g_cm3 <= 0:
        raise ValueError("density must be positive")
    return _REF_BOND_LENGTH * (_REF_DENSITY / density_g_cm3) ** (1.0 / 3.0)
