"""Ring statistics computation for atomic structures.

This module implements the Franzblau shortest-path ring algorithm for
analyzing ring structures in atomic networks.

Algorithm details, correctness invariants, and the CRN reference benchmark
are documented in docs/rings.md.

Reference:
    D.S. Franzblau, Phys. Rev. B 44, 4925 (1991)
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque
from ase import Atoms
from ase.neighborlist import neighbor_list

from glass.metrics.core import RingMetrics


def _build_seed_array(nat: int, i: np.ndarray) -> np.ndarray:
    """Build seed array for neighbor list traversal.
    
    The seed array indicates where each atom's neighbors start in the
    neighbors array. seed[k] is the index in the neighbors array where
    atom k's neighbors begin.
    
    Args:
        nat: Number of atoms
        i: Array of atom indices (source atoms in neighbor pairs)
        
    Returns:
        Seed array of length nat+1
    """
    seed = np.zeros(nat + 1, dtype=np.int64)
    
    # Count neighbors for each atom
    for atom_idx in i:
        seed[atom_idx + 1] += 1
    
    # Cumulative sum to get starting positions
    for k in range(1, nat + 1):
        seed[k] += seed[k - 1]
    
    return seed


def _compute_shortest_distances(
    nat: int,
    seed: np.ndarray,
    neighbors: np.ndarray,
    root: int,
) -> np.ndarray:
    """Compute shortest path distances from root atom to all others using BFS.
    
    Args:
        nat: Number of atoms
        seed: Seed array for neighbor list
        neighbors: Neighbor indices array
        root: Root atom to compute distances from
        
    Returns:
        Array of distances from root to each atom
    """
    dist = np.zeros(nat, dtype=np.int64)
    visited = np.zeros(nat, dtype=bool)
    
    # BFS from root
    queue = deque([root])
    visited[root] = True
    
    while queue:
        current = queue.popleft()
        current_dist = dist[current]
        
        # Iterate over neighbors
        for ni in range(seed[current], seed[current + 1]):
            j = neighbors[ni]
            if not visited[j]:
                visited[j] = True
                dist[j] = current_dist + 1
                queue.append(j)
    
    dist[root] = 0
    return dist


def _compute_distance_matrix(
    nat: int,
    seed: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Compute shortest path distance matrix between all pairs of atoms.
    
    Args:
        nat: Number of atoms
        seed: Seed array for neighbor list
        neighbors: Neighbor indices array
        
    Returns:
        Distance matrix of shape (nat, nat)
    """
    dist = np.zeros((nat, nat), dtype=np.int64)
    
    for root in range(nat):
        dist[root] = _compute_shortest_distances(nat, seed, neighbors, root)
    
    return dist


def _normsq(v: np.ndarray) -> float:
    """Compute squared norm of a vector."""
    return float(np.dot(v, v))


class _Walker:
    """Walker for ring detection algorithm.

    Phase (away vs. closer) is carried as an explicit string flag
    instead of being encoded in the sign of `vertex`. The original encoding
    (`vertex > 0` == away, negative == closer) is ambiguous for atom index 0,
    since `-0 == 0` in both Python and numpy int arithmetic: a walker that
    steps onto atom 0 while still in the "away" phase gets silently
    misrouted to `_step_closer` on the next iteration. `vertex` here always
    holds the true non-negative atom index.
    """

    __slots__ = ['vertex', 'phase', 'previous_vertex', 'ring_vertices', 'distances_to_root']

    def __init__(
        self,
        vertex: int,
        previous_vertex: int,
        step_dist: np.ndarray,
        phase: str = 'away',
    ):
        self.vertex = vertex
        self.phase = phase
        self.previous_vertex = previous_vertex
        self.ring_vertices = [vertex]
        self.distances_to_root = [step_dist.copy()]

    def copy_with_step(
        self,
        new_vertex: int,
        step_dist: np.ndarray,
        phase: str,
    ) -> '_Walker':
        """Create a new walker by stepping to a new vertex."""
        new_walker = _Walker.__new__(_Walker)
        new_walker.vertex = new_vertex
        new_walker.phase = phase
        new_walker.previous_vertex = self.vertex
        new_walker.ring_vertices = self.ring_vertices + [new_vertex]

        # Add distance
        new_dist = self.distances_to_root[-1] + step_dist
        new_walker.distances_to_root = self.distances_to_root + [new_dist]

        return new_walker

    def ring_size(self) -> int:
        """Return the size of the ring found so far."""
        return len(self.ring_vertices)


def _step_away(
    walkers: list,
    walker: _Walker,
    root: int,
    nat: int,
    seed: np.ndarray,
    neighbors: np.ndarray,
    r: np.ndarray,
    dist: np.ndarray,
    maxlength: int,
    done: np.ndarray,
) -> bool:
    """Step away from root vertex during ring search.
    
    Returns False if an error occurred.
    """
    i = walker.vertex
    
    for ni in range(seed[i], seed[i + 1]):
        j = neighbors[ni]
        
        # Check if edge has already been visited or if vertex is identical to
        # previous vertex of walker (this would be a reverse jump)
        if done[ni] or j == walker.previous_vertex:
            continue
        
        # Did we jump farther away from the root vertex?
        if dist[root, j] == dist[root, i] + 1:
            # Don't continue stepping further if we are already at half the
            # maximum ring length
            if maxlength < 0 or walker.ring_size() < (maxlength + 1) // 2:
                step_dist = r[ni]
                walkers.append(walker.copy_with_step(j, step_dist, phase='away'))
        # Did we either not change distance from root vertex or moved closer?
        elif dist[root, j] == dist[root, i] or dist[root, j] == dist[root, i] - 1:
            # This is a jump back towards the root
            step_dist = r[ni]
            new_walker = walker.copy_with_step(j, step_dist, phase='closer')
            walkers.append(new_walker)
        else:
            # Distance mismatch - should not happen in valid graph
            return False
    
    return True


def _step_closer(
    walkers: list,
    walker: _Walker,
    root: int,
    nat: int,
    seed: np.ndarray,
    neighbors: np.ndarray,
    r: np.ndarray,
    dist: np.ndarray,
    ringstat: dict,
    done: np.ndarray,
) -> bool:
    """Step closer to root vertex during ring search.
    
    Returns False if an error occurred.
    """
    i = walker.vertex

    for ni in range(seed[i], seed[i + 1]):
        j = neighbors[ni]
        
        # Check if edge has already been visited or if vertex is identical to
        # previous vertex of walker (this would be a reverse jump)
        if done[ni] or j == walker.previous_vertex:
            continue
        
        # Are we back to the root vertex?
        if j == root:
            # Check if the sum of bond vectors is zero. This is to
            # exclude chains that cross periodic boundaries.
            droot = walker.distances_to_root[-1]
            d = r[ni] + droot
            
            TOL = 0.0001
            if _normsq(d) < TOL:
                # Now we need to check whether this ring is SP
                is_sp = True
                
                # Add the root vertex
                ring_vertices = walker.ring_vertices + [root]
                ring_size = len(ring_vertices)
                
                # Check if the length along the ring agrees with the map
                # distance. Otherwise, there is a short ring that cuts this one.
                for m in range(ring_size):
                    for n in range(m + 1, ring_size):
                        dn = n - m
                        if dn > ring_size // 2:
                            dn = ring_size - dn
                        
                        vm = ring_vertices[m]
                        vn = ring_vertices[n]
                        if dist[vm, vn] != dn:
                            is_sp = False
                            break
                    if not is_sp:
                        break
                
                if is_sp:
                    if ring_size not in ringstat:
                        ringstat[ring_size] = 0.0
                    ringstat[ring_size] += 1.0 / ring_size
        # Did we jump closer to the root vertex?
        elif dist[root, j] == dist[root, i] - 1:
            step_dist = r[ni]
            walkers.append(walker.copy_with_step(j, step_dist, phase='closer'))
        # We discard this path if we jump away again
    
    return True


def _find_sp_rings(
    nat: int,
    seed: np.ndarray,
    neighbors: np.ndarray,
    r: np.ndarray,
    dist: np.ndarray,
    maxlength: int,
) -> dict:
    """Find shortest-path rings in the atomic network.
    
    Args:
        nat: Number of atoms
        seed: Seed array for neighbor list
        neighbors: Neighbor indices array
        r: Bond vectors array (nneigh, 3)
        dist: Distance matrix
        maxlength: Maximum ring length to consider
        
    Returns:
        Dictionary mapping ring size to count
    """
    ringstat = {}
    nneigh = len(neighbors)
    
    # Loop over all vertices
    for a in range(nat):
        # Loop over neighbours of vertex a, i.e. walk on graph
        for na in range(seed[a], seed[a + 1]):
            b = neighbors[na]
            
            # Only walk in one direction
            if a >= b:
                continue
            
            # Create done array for this starting edge
            # We have visited this site. Mark edge and its reverse as visited.
            done = np.zeros(nneigh, dtype=bool)
            done[na] = True
            
            # Mark reverse edge as visited
            for ni in range(seed[b], seed[b + 1]):
                if neighbors[ni] == a:
                    done[ni] = True
                    break
            
            # Initialize walker on atom b coming from atom a
            initial_dist = r[na]
            walkers = [_Walker(b, a, initial_dist, phase='away')]

            # Continue loop while there are walkers active
            while walkers:
                new_walkers = []

                # Loop over all walkers and advance them
                for walker in walkers:
                    # Walker walks away from root
                    if walker.phase == 'away':
                        if not _step_away(
                            new_walkers, walker, a, nat, seed, neighbors, r, dist, maxlength, done
                        ):
                            return {}
                    # Walker walks towards root
                    else:
                        if not _step_closer(
                            new_walkers, walker, a, nat, seed, neighbors, r, dist, ringstat, done
                        ):
                            return {}
                
                # Copy new walker list to old walker list
                walkers = new_walkers
    
    return ringstat


def _resolve_engine(engine: str) -> str:
    """Resolve 'auto' → 'numba' if available, else 'python'."""
    if engine == "auto":
        try:
            import numba  # noqa: F401
            return "numba"
        except ImportError:
            return "python"
    if engine not in ("python", "numba"):
        raise ValueError(f"Unknown engine: {engine!r}; expected 'python', 'numba', or 'auto'")
    return engine


def compute_rings(
    atoms: Atoms,
    cutoff: Optional[float] = None,
    maxlength: int = 10,
    auto_cutoff: bool = True,
    engine: str = "auto",
) -> RingMetrics:
    """Compute shortest-path ring statistics for atomic structures.

    Implements the Franzblau algorithm for finding shortest-path rings
    in atomic networks. This is useful for characterizing the topology
    of amorphous materials.

    Reference:
        D.S. Franzblau, Phys. Rev. B 44, 4925 (1991)

    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff for neighbor identification (Angstrom). If None and
            auto_cutoff=True, uses PDF coordination cutoff.
        maxlength: Maximum ring size to consider
        auto_cutoff: If True and cutoff is None, automatically determine
            cutoff from PDF first minimum
        engine: 'auto' (default; numba if installed, else python),
            'numba', or 'python'.

    Returns:
        RingMetrics object with ring statistics
    """
    engine = _resolve_engine(engine)
    # Determine cutoff
    if cutoff is None:
        if auto_cutoff:
            from glass.metrics.structural import compute_pdf
            pdf_metrics = compute_pdf(atoms, cutoff=8.0, bin_size=100)
            cutoff = pdf_metrics.coord_cutoff
            if cutoff is None:
                cutoff = 3.0  # Default for Si
        else:
            cutoff = 3.0
    
    # Get neighbor list using ASE
    i, j, D = neighbor_list('ijD', atoms, cutoff)
    
    # Remove self-interactions
    mask = i != j
    i = i[mask]
    j = j[mask]
    D = D[mask]
    
    if len(i) == 0:
        # No neighbors found
        ring_lengths = np.arange(0, maxlength + 1)
        return RingMetrics(
            ring_lengths=ring_lengths,
            ring_counts=np.zeros(maxlength + 1, dtype=np.float64),
            ring_fractions=np.zeros(maxlength + 1),
            total_rings=0.0,
            cutoff=cutoff,
            maxlength=maxlength,
        )
    
    # Number of atoms
    nat = len(atoms)
    
    # Build seed array and neighbor list
    seed = _build_seed_array(nat, i)
    
    # Reorganize neighbors to match seed array order
    # Sort by source atom i
    order = np.argsort(i, kind='stable')
    neighbors = j[order]
    r = D[order]
    
    ring_lengths = np.arange(0, maxlength + 1)

    if engine == "numba":
        from glass.metrics import _rings_numba as _nb
        seed_i32 = seed.astype(np.int32)
        neighbors_i32 = neighbors.astype(np.int32)
        r_f64 = r.astype(np.float64)
        dist = _nb.bfs_dist_matrix(seed_i32, neighbors_i32, nat)
        ring_counts = _nb.find_sp_rings(seed_i32, neighbors_i32, r_f64, dist, maxlength)
    else:
        dist = _compute_distance_matrix(nat, seed, neighbors)
        ringstat = _find_sp_rings(nat, seed, neighbors, r, dist, maxlength)
        ring_counts = np.zeros(maxlength + 1, dtype=np.float64)
        for size, count in ringstat.items():
            if 0 <= size <= maxlength:
                ring_counts[size] = count
    
    # Compute fractions
    total = np.sum(ring_counts)
    if total > 0:
        ring_fractions = ring_counts / total * 100
    else:
        ring_fractions = np.zeros_like(ring_counts, dtype=float)
    
    return RingMetrics(
        ring_lengths=ring_lengths,
        ring_counts=ring_counts,
        ring_fractions=ring_fractions,
        total_rings=float(total),
        cutoff=cutoff,
        maxlength=maxlength,
    )


def compute_rings_distribution(
    atoms_list,
    cutoff: Optional[float] = None,
    maxlength: int = 10,
    auto_cutoff: bool = True,
    engine: str = "auto",
) -> RingMetrics:
    """Compute ring statistics averaged over multiple structures.

    Args:
        atoms_list: List of ASE Atoms objects or a single Atoms object
        cutoff: Cutoff for neighbor identification
        maxlength: Maximum ring size to consider
        auto_cutoff: If True, automatically determine cutoff
        engine: 'auto' (default), 'numba', or 'python'.

    Returns:
        RingMetrics with averaged statistics
    """
    # Ensure list
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    
    nframes = len(atoms_list)
    
    # Accumulate counts
    total_counts = np.zeros(maxlength + 1, dtype=np.float64)
    total_fractions = np.zeros(maxlength + 1, dtype=np.float64)
    
    for atoms in atoms_list:
        metrics = compute_rings(atoms, cutoff=cutoff, maxlength=maxlength,
                                auto_cutoff=auto_cutoff, engine=engine)
        total_counts += metrics.ring_counts
        total_fractions += metrics.ring_fractions
    
    # Average
    avg_counts = total_counts / nframes
    avg_fractions = total_fractions / nframes
    
    ring_lengths = np.arange(0, maxlength + 1)
    
    return RingMetrics(
        ring_lengths=ring_lengths,
        ring_counts=avg_counts,
        ring_fractions=avg_fractions,
        total_rings=float(np.sum(avg_counts)),
        cutoff=cutoff if cutoff else 0.0,
        maxlength=maxlength,
    )
