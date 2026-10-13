"""Time-step schedules for reverse-SDE integration.

``power_law_ts`` generalises ``torch.linspace(tmax, tmin, tstep)`` with a
concentration exponent ``rho``:

    t_i = tmin + (tmax - tmin) * ((N - i) / N) ** rho,   i = 0..N-1

``rho = 1`` reproduces the linspace trajectory. ``rho > 1`` concentrates the
steps near ``t = 0`` (where bonds crystallise), ``rho < 1`` pushes them toward
``t = tmax``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def power_law_ts(
    tmin: float,
    tmax: float,
    tstep: int,
    rho: float = 1.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a descending trajectory ``t_0 = tmax, ..., t_{N-1} = tmin``.

    Args:
        tmin: Final (smallest) time.
        tmax: Initial (largest) time.
        tstep: Number of steps (``N``).
        rho: Concentration exponent. ``rho = 1`` is linspace;
            ``rho > 1`` concentrates near ``tmin``; ``rho < 1`` near ``tmax``.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``(tstep,)``, monotonically decreasing.
    """
    if tstep < 2:
        raise ValueError(f"tstep must be >= 2, got {tstep}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")

    N = tstep - 1
    i = torch.arange(tstep, device=device, dtype=dtype)
    frac = (N - i) / N
    ts = tmin + (tmax - tmin) * frac.pow(rho)
    return ts


def linear_ts(
    tmin: float,
    tmax: float,
    tstep: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convenience alias for ``power_law_ts`` with ``rho = 1``."""
    return power_law_ts(tmin, tmax, tstep, rho=1.0, device=device, dtype=dtype)
