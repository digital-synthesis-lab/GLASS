"""Short end-to-end smoke test for the WWW driver."""

from __future__ import annotations

import numpy as np

from glass.algorithms.crn.www import generate_crn


def test_smoke_short_run() -> None:
    """Quick smoke test: N=64, 1 cycle, minimal anneal/quench."""
    net, stats = generate_crn(
        n_atoms=64,
        seed=0,
        n_cycles=1,
        n_anneal_per_atom=4,
        quench_attempts_per_atom=3,
        relax_local_steps=4,
        relax_full_max_iter=40,
    )
    assert net.n_atoms == 64
    assert np.all(net.degree == 4)
    assert net.bonds.shape == (128, 2)
    assert stats.final_energy <= stats.initial_energy + 1e-6
