import itertools

import torch


def _build_neighbors_dense(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
    shift_ints: torch.Tensor,
    shift_vecs: torch.Tensor,
):
    """Original (S, N, N, 3) enumeration. Fast for small N."""
    N = positions.shape[0]
    dtype = positions.dtype
    device = positions.device

    ri = positions.unsqueeze(0).unsqueeze(2)          # (1, N, 1, 3)
    rj = positions.unsqueeze(0).unsqueeze(1)          # (1, 1, N, 3)
    s = shift_vecs.unsqueeze(1).unsqueeze(1)          # (S, 1, 1, 3)
    disp = rj + s - ri                                # (S, N, N, 3)
    r2 = (disp * disp).sum(dim=-1)                    # (S, N, N)

    in_cut = r2 < cutoff * cutoff
    S = shift_ints.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    zero_shift_mask = (shift_ints == 0).all(dim=1)
    self_mask = eye & zero_shift_mask.view(S, 1, 1)
    in_cut = in_cut & (~self_mask)
    in_cut = in_cut & (r2 > 1e-24)

    s_idx, i_idx, j_idx = torch.nonzero(in_cut, as_tuple=True)
    shift_out = shift_vecs[s_idx]
    return i_idx, j_idx, shift_out


def _build_neighbors_cell_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
    shift_ints: torch.Tensor,
    shift_vecs: torch.Tensor,
):
    """Cell-list-based neighbour enumeration for large N.

    Wraps atoms into the primitive cell, buckets them by fractional
    coordinates on a uniform grid with bucket side ≥ cutoff along every
    axis, and enumerates pairs from each bucket against itself and its
    27-neighbour shell (combined with the supercell shift vectors). This
    avoids ever materialising an (S, N, N, 3) tensor.
    """
    N = positions.shape[0]
    dtype = positions.dtype
    device = positions.device

    cell_t = cell.to(dtype=dtype, device=device)
    cell_inv = torch.linalg.inv(cell_t)

    # Fractional coords in [0, 1). Detach is safe: this branch is used
    # only to decide which pairs to enumerate; the returned i/j indices
    # and shift vectors (plain tensors) carry no gradient information.
    with torch.no_grad():
        frac = positions.detach() @ cell_inv
        frac = frac - torch.floor(frac)
        # Clamp for float rounding (0.999999... * n -> n).
        frac = torch.clamp(frac, min=0.0, max=1.0 - 1e-12)

        # Bucket count per axis: each bucket's Cartesian extent along
        # axis a is |cell[a,:]| / n_a; we need that ≥ cutoff so the
        # 27-neighbour shell covers the interaction radius.
        axis_len = torch.linalg.norm(cell_t, dim=1)  # (3,)
        n_axis = torch.maximum(
            torch.floor(axis_len / cutoff).to(torch.long),
            torch.ones(3, dtype=torch.long, device=device),
        )
        nb = (frac * n_axis.to(dtype)).to(torch.long)
        nb = torch.minimum(nb, n_axis - 1)  # (N, 3)

        # Flatten 3D bucket indices.
        n0, n1, n2 = int(n_axis[0]), int(n_axis[1]), int(n_axis[2])
        bucket_flat = nb[:, 0] * (n1 * n2) + nb[:, 1] * n2 + nb[:, 2]  # (N,)

        # Sort atoms by bucket index so per-bucket slices are contiguous.
        order = torch.argsort(bucket_flat)
        bucket_sorted = bucket_flat[order]
        # Start/end offsets for each bucket (CSR-style).
        n_buckets = n0 * n1 * n2
        counts = torch.bincount(bucket_sorted, minlength=n_buckets)
        offsets = torch.empty(n_buckets + 1, dtype=torch.long, device=device)
        offsets[0] = 0
        torch.cumsum(counts, dim=0, out=offsets[1:])

    # Build the list of bucket offsets for the neighbour shell
    # (-1, 0, 1)^3 — 27 entries. Each shell offset also participates with
    # every supercell shift in shift_ints.
    shell_ints = torch.tensor(
        list(itertools.product((-1, 0, 1), repeat=3)),
        dtype=torch.long, device=device,
    )  # (27, 3)

    i_all, j_all, shift_all = [], [], []

    # Iterate over shell offsets + supercell shifts. For each combo,
    # enumerate all (source_bucket -> neighbour_bucket) pairs and
    # compute (r_j + shift - r_i) distances in chunks.
    with torch.no_grad():
        # Precompute a per-bucket owner list for convenience.
        # sorted_positions[k] belongs to bucket_sorted[k].
        pos_sorted = positions[order]

    # The result we ultimately return must reference original atom
    # indices (not the sorted order), so keep `order` handy.
    # We also need gradient-carrying positions by the ORIGINAL indices.
    pos_sorted_grad = positions.index_select(0, order)

    # Convert shift_ints to a Python list to iterate cheaply.
    shift_list = [tuple(int(v) for v in s) for s in shift_ints.tolist()]
    shell_list = [tuple(int(v) for v in s) for s in shell_ints.tolist()]

    # Precompute: for each bucket, 27 neighbour bucket indices (with
    # wrap). Because n_axis may be as small as 1, we compute modulo.
    # Build a (n_buckets, 27) table. Memory: n_buckets * 27 * 8 bytes.
    # For a 4000-atom 15 Å^3 cube with cutoff 3.2 that's ~64 buckets,
    # trivial. For a 50 Å^3 cube ~3500 buckets * 27 * 8 B = 760 KB.
    with torch.no_grad():
        bx = torch.arange(n0, device=device)
        by = torch.arange(n1, device=device)
        bz = torch.arange(n2, device=device)
        base = (
            bx.view(-1, 1, 1) * (n1 * n2)
            + by.view(1, -1, 1) * n2
            + bz.view(1, 1, -1)
        )  # (n0, n1, n2)
        base_flat = base.reshape(-1)  # (n_buckets,)

        # 27 neighbour bucket indices per source bucket.
        # shell_list (27, 3) -> deltas on (a,b,c).
        neigh_ids = []
        for dx, dy, dz in shell_list:
            nx = (bx.view(-1, 1, 1) + dx) % n0
            ny = (by.view(1, -1, 1) + dy) % n1
            nz = (bz.view(1, 1, -1) + dz) % n2
            nid = nx * (n1 * n2) + ny * n2 + nz
            neigh_ids.append(nid.reshape(-1))
        neigh_table = torch.stack(neigh_ids, dim=1)  # (n_buckets, 27)

    cutoff_sq = cutoff * cutoff
    order_np = order  # tensor

    # Process one supercell shift at a time. For each shift, enumerate
    # for each bucket its 27 neighbour buckets in the target image, and
    # collect atom-pairs with squared distance < cutoff.
    zero_shift_idx = None
    for s_k, (sx, sy, sz) in enumerate(shift_list):
        if sx == 0 and sy == 0 and sz == 0:
            zero_shift_idx = s_k
            break

    for s_k, (sx, sy, sz) in enumerate(shift_list):
        shift_cart = shift_vecs[s_k]  # (3,)

        # For this supercell shift, the "target" of a bucket's
        # 27-shell lookup is the same 27 buckets in modular arithmetic;
        # the shift is added to the atom positions.
        for shell_k, (dx, dy, dz) in enumerate(shell_list):
            # Flat (source_bucket -> neighbour_bucket) map for this
            # shell offset.
            # We process source atoms in a single flat vector and find
            # their matched neighbour bucket atoms.

            # source atoms: all atoms (in sorted order). For each atom
            # k_src (in sorted order), its source bucket is
            # bucket_sorted[k_src]; its neighbour bucket is neigh_table[
            # source_bucket, shell_k].
            src_bucket = bucket_sorted  # (N,)
            dst_bucket = neigh_table[src_bucket, shell_k]  # (N,)

            dst_start = offsets[dst_bucket]       # (N,)
            dst_count = counts[dst_bucket]        # (N,)

            # For each source atom k_src, we need to enumerate
            # dst_count[k_src] neighbour indices. The total number of
            # pairs for this (shift, shell) combo is dst_count.sum().
            total_pairs = int(dst_count.sum().item())
            if total_pairs == 0:
                continue

            # Build per-pair (src index, dst offset within bucket).
            # src_rep repeats each src atom dst_count[k_src] times.
            src_rep = torch.repeat_interleave(
                torch.arange(N, device=device), dst_count
            )
            # dst offsets: for each source, 0 .. dst_count[src]-1. Build
            # by subtracting a running start from arange(total_pairs).
            pair_start = torch.repeat_interleave(
                torch.cumsum(dst_count, dim=0) - dst_count, dst_count
            )
            within = torch.arange(total_pairs, device=device) - pair_start
            dst_rep_start = torch.repeat_interleave(dst_start, dst_count)
            dst_rep = dst_rep_start + within  # indices into sorted arrays (N,)

            # Compute displacements in chunks to bound memory.
            CHUNK = 1_048_576  # 1M pairs per chunk -> 12 MB float32 for disp
            for start in range(0, total_pairs, CHUNK):
                stop = min(start + CHUNK, total_pairs)
                s_slice = src_rep[start:stop]
                d_slice = dst_rep[start:stop]
                ri_c = pos_sorted_grad[s_slice]
                rj_c = pos_sorted_grad[d_slice]
                disp = rj_c + shift_cart - ri_c
                r2 = (disp * disp).sum(dim=-1)
                mask = (r2 < cutoff_sq) & (r2 > 1e-24)
                if s_k == zero_shift_idx:
                    # Exclude i == j at zero shift. d_slice and s_slice
                    # are in sorted-array space; they're equal iff the
                    # original atoms are the same, which happens when
                    # the source bucket's 27-shell lands on itself and
                    # the atom picks itself.
                    mask = mask & (s_slice != d_slice)
                if mask.any():
                    sel = mask.nonzero(as_tuple=True)[0]
                    # Translate sorted-order indices back to original
                    # atom indices.
                    i_all.append(order[s_slice[sel]])
                    j_all.append(order[d_slice[sel]])
                    shift_all.append(
                        shift_cart.unsqueeze(0).expand(sel.numel(), 3).contiguous()
                    )

    if not i_all:
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_s = torch.empty(0, 3, dtype=dtype, device=device)
        return empty_i, empty_i, empty_s

    i_out = torch.cat(i_all)
    j_out = torch.cat(j_all)
    shift_out = torch.cat(shift_all)
    return i_out, j_out, shift_out


def build_neighbors(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
):
    """Enumerate directed neighbour pairs (i, j, shift) with |r_ij| < cutoff.

    Parameters
    ----------
    positions : (N, 3) float tensor
    cell : (3, 3) float tensor
    pbc : iterable of 3 bools
    cutoff : float

    Returns
    -------
    i_idx, j_idx : (M,) long tensors
    shift_vec : (M, 3) float tensor of cartesian shifts (= S @ cell)
        so that r_j_image = positions[j] + shift_vec, and
        r_ij_vec = positions[j] + shift_vec - positions[i].

    For small systems (N <= 512) the dense (S, N, N) enumeration is
    used. For larger systems we switch to a cell-list path that is
    O(N · max_neighbours) rather than O(N²), avoiding OOM on realistic
    cells.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    pbc = tuple(bool(p) for p in pbc)

    # Enumerate supercell shifts only along PBC directions.
    ranges = [(-1, 0, 1) if p else (0,) for p in pbc]
    shift_ints = torch.tensor(
        list(itertools.product(*ranges)), dtype=torch.long, device=device
    )  # (S, 3)
    cell_t = cell.to(dtype=dtype, device=device)
    shift_vecs = shift_ints.to(dtype=dtype) @ cell_t  # (S, 3)

    # Heuristic: switch to cell-list at moderate N. The dense path
    # allocates ~S·N²·3·sizeof(dtype); at N=512 and float64 that's
    # ~1.7 GiB worst case (S=27), which is already tight. For N ≤ 256
    # the dense path is ~0.4 GiB and its faster vectorisation wins.
    use_cell_list = (N > 256) and any(pbc)

    if use_cell_list:
        return _build_neighbors_cell_list(
            positions, cell_t, pbc, cutoff, shift_ints, shift_vecs
        )
    return _build_neighbors_dense(
        positions, cell_t, pbc, cutoff, shift_ints, shift_vecs
    )
