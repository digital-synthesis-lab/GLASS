"""Numba-accelerated Franzblau shortest-path ring search.

This is the optimised twin of the pure-Python reference implementation in
``glass.metrics.rings``. Algorithmic semantics, including the two
correctness invariants (divide each detection by ring size; expansion gate
``(maxlength + 1) // 2``), are identical. See ``docs/rings.md``.

The original implementation encoded walker phase (away/closer) in
the SIGN of ``vertex`` (away: vertex>0; closer: vertex<0, decoded via
``i = -v``). Because ``-0 == 0``, a walker that steps onto atom index 0
while still in the "away" phase is misrouted to the "closer" branch on the
next iteration, silently killing some ring detections that pass through
atom 0. Phase is carried here as an explicit ``cur_phase``/``new_phase``
array (1=away, 0=closer) instead; ``vertex`` always holds the true
non-negative atom index. Everything else is unchanged.

The kernel is parallel: the outer loop over starting edges is
``numba.prange`` and each iteration writes into thread-local scratch.
Use ``numba.set_num_threads(N)`` to control concurrency.
"""

import numpy as np
import numba
from numba import njit, prange


# Generous upper bound on the number of walkers active at any one step
# during a single starting-edge walk. For coordination 4 and maxlength 12
# the worst-case frontier is ~3**6 = 729; 4096 leaves comfortable headroom.
MAX_WALKERS = 4096

# Closure-vector squared-norm tolerance for accepting a ring (rejects rings
# that close only via a periodic image). Matches the Python reference.
_CLOSURE_TOL_SQ = 1e-4


@njit(cache=True, parallel=True)
def bfs_dist_matrix(seed, neighbors, nat):
    """All-pairs BFS distance matrix. Each row is computed independently
    in parallel via ``prange``. Returns int32[nat, nat]."""
    dist = np.full((nat, nat), 0, dtype=np.int32)
    for root in prange(nat):
        visited = np.zeros(nat, dtype=np.bool_)
        queue = np.empty(nat, dtype=np.int32)
        queue[0] = root
        visited[root] = True
        head = 0
        tail = 1
        while head < tail:
            current = queue[head]
            head += 1
            curr_dist = dist[root, current]
            for ni in range(seed[current], seed[current + 1]):
                j = neighbors[ni]
                if not visited[j]:
                    visited[j] = True
                    dist[root, j] = curr_dist + 1
                    queue[tail] = j
                    tail += 1
    return dist


@njit(cache=True, inline="always")
def _reverse_edge_index(seed, neighbors, a, b, na):
    """Index in ``neighbors`` of the reverse edge (b → a) corresponding to
    the directed edge ``na`` (a → b). Returns -1 if not found (shouldn't
    happen for a well-formed neighbour list)."""
    for ni in range(seed[b], seed[b + 1]):
        if neighbors[ni] == a:
            return ni
    return -1


@njit(cache=True)
def _walk_one_edge(
    a, b, na, na_rev,
    seed, neighbors, r, dist, maxlength,
    done,                                            # bool[nneigh]
    cur_vertex, cur_phase, cur_prev, cur_size, cur_ring, cur_accum,
    new_vertex, new_phase, new_prev, new_size, new_ring, new_accum,
    out_counts,                                      # float64[maxlength+1]
):
    """Run the Franzblau walker for the single starting edge a → b and
    accumulate detected SP-ring contributions (1 / ring_size each) into
    ``out_counts``. Caller supplies all scratch buffers.

    Phase (1=away, 0=closer) is an explicit array instead of being
    encoded in the sign of ``vertex`` -- see module docstring."""

    nneigh = neighbors.shape[0]

    # Reset done and mark the starting edge + its reverse as visited.
    for k in range(nneigh):
        done[k] = False
    done[na] = True
    if na_rev >= 0:
        done[na_rev] = True

    # Initial walker: at b, came from a.
    cur_n = 1
    cur_vertex[0] = b
    cur_phase[0] = 1
    cur_prev[0] = a
    cur_size[0] = 1
    cur_ring[0, 0] = b
    cur_accum[0, 0] = r[na, 0]
    cur_accum[0, 1] = r[na, 1]
    cur_accum[0, 2] = r[na, 2]

    half_gate = (maxlength + 1) // 2

    while cur_n > 0:
        new_n = 0
        for w in range(cur_n):
            v = cur_vertex[w]
            phase = cur_phase[w]
            prev = cur_prev[w]
            sz = cur_size[w]

            if phase == 1:
                # ---- step away ----
                i = v
                d_ri = dist[a, i]
                for ni in range(seed[i], seed[i + 1]):
                    j = neighbors[ni]
                    if done[ni] or j == prev:
                        continue
                    d_rj = dist[a, j]
                    if d_rj == d_ri + 1:
                        # Further from root: only continue if we haven't
                        # exceeded the half-ring expansion gate.
                        if maxlength < 0 or sz < half_gate:
                            if new_n >= MAX_WALKERS:
                                return  # frontier overflow: silently bail
                            new_vertex[new_n] = j
                            new_phase[new_n] = 1
                            new_prev[new_n] = i
                            new_size[new_n] = sz + 1
                            for k in range(sz):
                                new_ring[new_n, k] = cur_ring[w, k]
                            new_ring[new_n, sz] = j
                            new_accum[new_n, 0] = cur_accum[w, 0] + r[ni, 0]
                            new_accum[new_n, 1] = cur_accum[w, 1] + r[ni, 1]
                            new_accum[new_n, 2] = cur_accum[w, 2] + r[ni, 2]
                            new_n += 1
                    elif d_rj == d_ri or d_rj == d_ri - 1:
                        # Turn back: switch to closer phase (explicit flag).
                        if new_n >= MAX_WALKERS:
                            return
                        new_vertex[new_n] = j
                        new_phase[new_n] = 0
                        new_prev[new_n] = i
                        new_size[new_n] = sz + 1
                        for k in range(sz):
                            new_ring[new_n, k] = cur_ring[w, k]
                        new_ring[new_n, sz] = j
                        new_accum[new_n, 0] = cur_accum[w, 0] + r[ni, 0]
                        new_accum[new_n, 1] = cur_accum[w, 1] + r[ni, 1]
                        new_accum[new_n, 2] = cur_accum[w, 2] + r[ni, 2]
                        new_n += 1
                    # else: distance mismatch — discard (matches Python ref).
            else:
                # ---- step closer ----
                i = v
                d_ri = dist[a, i]
                for ni in range(seed[i], seed[i + 1]):
                    j = neighbors[ni]
                    if done[ni] or j == prev:
                        continue
                    if j == a:
                        # Candidate closure: check periodic-vector sum.
                        dx = cur_accum[w, 0] + r[ni, 0]
                        dy = cur_accum[w, 1] + r[ni, 1]
                        dz = cur_accum[w, 2] + r[ni, 2]
                        if dx * dx + dy * dy + dz * dz >= _CLOSURE_TOL_SQ:
                            continue

                        ring_size = sz + 1  # include root
                        if ring_size > maxlength:
                            continue

                        # SP test: every pair (m, n) on the ring must have
                        # ring-distance == graph-distance.
                        is_sp = True
                        for m in range(ring_size):
                            if m < sz:
                                vm = cur_ring[w, m]
                                if vm < 0:
                                    vm = -vm
                            else:
                                vm = a
                            for n in range(m + 1, ring_size):
                                if n < sz:
                                    vn = cur_ring[w, n]
                                    if vn < 0:
                                        vn = -vn
                                else:
                                    vn = a
                                dn = n - m
                                if dn > ring_size // 2:
                                    dn = ring_size - dn
                                if dist[vm, vn] != dn:
                                    is_sp = False
                                    break
                            if not is_sp:
                                break

                        if is_sp:
                            out_counts[ring_size] += 1.0 / ring_size
                    elif dist[a, j] == d_ri - 1:
                        if new_n >= MAX_WALKERS:
                            return
                        new_vertex[new_n] = j
                        new_phase[new_n] = 0
                        new_prev[new_n] = i
                        new_size[new_n] = sz + 1
                        for k in range(sz):
                            new_ring[new_n, k] = cur_ring[w, k]
                        new_ring[new_n, sz] = j
                        new_accum[new_n, 0] = cur_accum[w, 0] + r[ni, 0]
                        new_accum[new_n, 1] = cur_accum[w, 1] + r[ni, 1]
                        new_accum[new_n, 2] = cur_accum[w, 2] + r[ni, 2]
                        new_n += 1
                    # else: walker would jump away again — discard.

        # Promote new frontier to current (in-place copy; frontier sizes
        # are small so this is cheap).
        cur_n = new_n
        for w in range(new_n):
            cur_vertex[w] = new_vertex[w]
            cur_phase[w] = new_phase[w]
            cur_prev[w] = new_prev[w]
            cur_size[w] = new_size[w]
            for k in range(new_size[w]):
                cur_ring[w, k] = new_ring[w, k]
            cur_accum[w, 0] = new_accum[w, 0]
            cur_accum[w, 1] = new_accum[w, 1]
            cur_accum[w, 2] = new_accum[w, 2]


@njit(parallel=True)
def find_sp_rings(seed, neighbors, r, dist, maxlength):
    """Numba parallel driver. Returns float64[maxlength+1] ring counts."""
    nat = seed.shape[0] - 1
    nneigh = neighbors.shape[0]
    ring_buf_w = maxlength + 2

    # Build the list of starting-edge indices (a < b) and remember 'a' for each.
    # Pass 1: count.
    n_edges = 0
    for a in range(nat):
        for na in range(seed[a], seed[a + 1]):
            if a < neighbors[na]:
                n_edges += 1
    edge_na = np.empty(n_edges, dtype=np.int32)
    edge_a = np.empty(n_edges, dtype=np.int32)
    ei = 0
    for a in range(nat):
        for na in range(seed[a], seed[a + 1]):
            if a < neighbors[na]:
                edge_na[ei] = na
                edge_a[ei] = a
                ei += 1

    # Thread-local scratch.
    n_threads = numba.get_num_threads()
    done_tls = np.zeros((n_threads, nneigh), dtype=np.bool_)
    cur_vertex = np.zeros((n_threads, MAX_WALKERS), dtype=np.int32)
    cur_phase = np.zeros((n_threads, MAX_WALKERS), dtype=np.int8)
    cur_prev = np.zeros((n_threads, MAX_WALKERS), dtype=np.int32)
    cur_size = np.zeros((n_threads, MAX_WALKERS), dtype=np.int32)
    cur_ring = np.zeros((n_threads, MAX_WALKERS, ring_buf_w), dtype=np.int32)
    cur_accum = np.zeros((n_threads, MAX_WALKERS, 3), dtype=np.float64)
    new_vertex = np.zeros((n_threads, MAX_WALKERS), dtype=np.int32)
    new_phase = np.zeros((n_threads, MAX_WALKERS), dtype=np.int8)
    new_prev = np.zeros((n_threads, MAX_WALKERS), dtype=np.int32)
    new_size = np.zeros((n_threads, MAX_WALKERS), dtype=np.int32)
    new_ring = np.zeros((n_threads, MAX_WALKERS, ring_buf_w), dtype=np.int32)
    new_accum = np.zeros((n_threads, MAX_WALKERS, 3), dtype=np.float64)
    counts_tls = np.zeros((n_threads, maxlength + 1), dtype=np.float64)

    for ei in prange(n_edges):
        na = edge_na[ei]
        a = edge_a[ei]
        b = neighbors[na]
        na_rev = _reverse_edge_index(seed, neighbors, a, b, na)
        tid = numba.get_thread_id()
        _walk_one_edge(
            a, b, na, na_rev,
            seed, neighbors, r, dist, maxlength,
            done_tls[tid],
            cur_vertex[tid], cur_phase[tid], cur_prev[tid], cur_size[tid], cur_ring[tid], cur_accum[tid],
            new_vertex[tid], new_phase[tid], new_prev[tid], new_size[tid], new_ring[tid], new_accum[tid],
            counts_tls[tid],
        )

    # Reduce across threads.
    out = np.zeros(maxlength + 1, dtype=np.float64)
    for t in range(n_threads):
        for k in range(maxlength + 1):
            out[k] += counts_tls[t, k]
    return out
