"""Guidance model factory and utilities for conditional generation.

This module provides factory functions for creating guidance models and loading
experimental data for conditional structure generation.
"""

import json
import math
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import torch
from torch import Tensor

from glass.lit.modules import (
    DifferentiableRDF,
    DifferentiableADF,
    DifferentiableXRD,
    DifferentiableND,
    LitSpecNet,
)


def _sorted_by_z(element_names) -> list:
    """Sort element symbols by atomic number.

    The one-hot ``species`` columns produced by sklearn's OneHotEncoder are
    sorted by atomic number, so XRD/ND per-element tables must match that
    order (e.g. SiO2 -> ["O", "Si"]).
    """
    from ase.data import atomic_numbers

    return sorted(element_names, key=lambda el: atomic_numbers[el])


def create_guidance_model(
    guidance_type: str,
    device: Union[str, torch.device],
    cutoff: float,
    bin_size: int = 100,
    angle_bins: int = 100,
    adf_sigma: float = 0.1,
    adf_cutoff: float = 3.5,
    element_names: Optional[List[str]] = None,
    qmin: float = 1.0,
    qmax: float = 20.0,
    qstep: float = 0.1,
    biso: float = 1.5,
    spec_model_path: Optional[Union[str, Path]] = None,
    pdf_cutoff: Optional[float] = None,
):
    """Create a guidance model for conditional generation.
    
    Args:
        guidance_type: Type of guidance (pdf, adf, xrd, nd, exafs, xanes)
        device: Torch device
        cutoff: Graph cutoff radius
        bin_size: Number of bins for PDF
        angle_bins: Number of angle bins for ADF
        adf_sigma: Gaussian kernel width for ADF
        adf_cutoff: Cutoff radius for ADF triplet search
        element_names: List of element names for XRD/ND
        qmin: Minimum q value for XRD/ND
        qmax: Maximum q value for XRD/ND
        qstep: Q step size for XRD/ND
        biso: Debye-Waller B factor for XRD/ND
        spec_model_path: Path to spectral model checkpoint for EXAFS/XANES
        pdf_cutoff: Maximum r of the PDF (default: ``cutoff``)
    
    Returns:
        Guidance model ready for inference
    
    Raises:
        ValueError: If required parameters are missing for the guidance type
    """
    device = torch.device(device)
    
    if guidance_type == "pdf":
        model = DifferentiableRDF(
            cutoff=pdf_cutoff if pdf_cutoff else cutoff, bin_size=bin_size, sigma=0.15
        )
        model.eval()
        return model
    
    elif guidance_type == "adf":
        model = DifferentiableADF(
            cutoff=adf_cutoff,
            angle_bins=angle_bins,
            angle_range=[0, math.pi],
            sigma=adf_sigma,
            # Compare the shape of the distribution. Without normalization the
            # number of triplets cannot be changed through the angles, and the
            # only way to lower the histogram is to push angles off the grid
            # towards 0, which collapses atoms onto each other.
            normalize=True,
        )
        model.eval()
        return model
    
    elif guidance_type == "xrd":
        if not element_names:
            raise ValueError("element_names required for xrd guidance")
        model = DifferentiableXRD(
            q_vals=[qmin, qmax, qstep],
            element_names=_sorted_by_z(element_names),
            biso=biso,
        )
        model.to(device).eval()
        return model
    
    elif guidance_type == "nd":
        if not element_names:
            raise ValueError("element_names required for nd guidance")
        model = DifferentiableND(
            q_vals=[qmin, qmax, qstep],
            element_names=_sorted_by_z(element_names),
            biso=biso,
        )
        model.to(device).eval()
        return model
    
    elif guidance_type in ("exafs", "xanes"):
        if not spec_model_path:
            raise ValueError(f"spec_model_path required for {guidance_type} guidance")
        spec_net = LitSpecNet.load_from_checkpoint(spec_model_path, map_location=device)
        spec_net.eval()
        spec_net.ema_model.to(device).eval()
        return spec_net.ema_model
    
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")


def load_experimental_data(
    exp_data_path: Union[str, Path],
    guidance_type: str,
    cutoff: float,
    bin_size: int,
    angle_bins: int,
    qmin: float,
    qmax: float,
    qstep: float,
    device: Union[str, torch.device],
) -> Tensor:
    """Load and interpolate experimental data.
    
    Args:
        exp_data_path: Path to JSON file with experimental data
        guidance_type: Type of guidance (determines interpolation grid)
        cutoff: Graph cutoff radius
        bin_size: Number of bins for PDF
        angle_bins: Number of angle bins for ADF
        qmin: Minimum q value for XRD/ND
        qmax: Maximum q value for XRD/ND
        qstep: Q step size for XRD/ND
        device: Torch device
    
    Returns:
        Interpolated target tensor on specified device
    
    Raises:
        ValueError: If guidance type is not supported for experimental data
    """
    device = torch.device(device)
    
    with open(exp_data_path) as f:
        d = json.load(f)
    
    y_exp = np.array(d.get("y", d.get("g", d.get("g_r"))))
    if guidance_type in ("exafs", "xanes"):
        # Atom-averaged spectrum on the grid of the surrogate, used as is
        return torch.from_numpy(y_exp).float().reshape(1, -1).to(device)
    x_exp = np.array(d.get("x", d.get("r")))
    
    if guidance_type == "pdf":
        bin_edges = np.linspace(0, cutoff, bin_size + 1)
        x_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    elif guidance_type == "adf":
        x_grid = np.linspace(0, math.pi, angle_bins)
    elif guidance_type in ("xrd", "nd"):
        x_grid = np.arange(qmin, qmax, qstep)
    else:
        raise ValueError(
            f"Experimental data not yet supported for {guidance_type}"
        )
    
    y_interp = np.interp(x_grid, x_exp, y_exp)
    return torch.from_numpy(y_interp).float().unsqueeze(0).to(device)
