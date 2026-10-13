"""Likelihood score computation for guided structure generation.

This module provides the LikelihoodScore class for computing guidance gradients
in conditional structure generation.
"""

import torch
from torch import nn, Tensor
from typing import Tuple

from glass.nn import periodic_radius_graph


class LikelihoodScore(nn.Module):
    """Compute likelihood score for conditional generation.
    
    This class computes the gradient of the mismatch between predicted and target
    features (e.g., PDF, XRD, EXAFS) to guide the generation process.
    
    Args:
        score_net: Score network (EMA model)
        guidance_model: Guidance model for computing features
        target_y: Target feature values
        rho: Guidance strength
        diffuser: Diffuser object for sigma computation
        guidance_type: Type of guidance (pdf, adf, xrd, nd, exafs, xanes)
        cutoff: Graph cutoff radius
        spec_average: [exafs/xanes] Compare the spectrum averaged over all
            atoms with a target of shape (1, M), instead of comparing the
            spectra of atom i in the generated and reference structures.
    """
    
    def __init__(
        self,
        score_net,
        guidance_model,
        target_y: Tensor,
        rho: float,
        diffuser,
        guidance_type: str,
        cutoff: float,
        spec_average: bool = False,
    ):
        super().__init__()
        self.score_net = score_net
        self.guidance_model = guidance_model
        self.target_y = target_y
        self.rho = rho
        self.diffuser = diffuser
        self.guidance_type = guidance_type
        self.cutoff = cutoff
        self.spec_average = spec_average
    
    def forward(
        self,
        species: Tensor,
        pos: Tensor,
        cell: Tensor,
        t: Tensor,
        cut: float,
    ) -> Tuple[Tensor, Tensor]:
        """Compute likelihood score and norm.
        
        Args:
            species: Atomic species tensor
            pos: Atomic positions tensor
            cell: Unit cell tensor
            t: Time step
            cut: Graph cutoff
        
        Returns:
            (likelihood_score, norm) where norm is the feature mismatch
        """
        with torch.enable_grad():
            pos = pos.detach().clone().requires_grad_(True)
            edge_index, edge_vec = periodic_radius_graph(pos, cut, cell)
            edge_attr = torch.hstack(
                [edge_vec, edge_vec.norm(dim=-1, keepdim=True)]
            )
            sigma = self.diffuser.sigma(t)
            
            with torch.no_grad():
                score = self.score_net(species, edge_index, edge_attr, t, sigma)
            est_clean_pos = pos + sigma.pow(2) * score
            
            if self.guidance_type in ("pdf", "adf"):
                pred_y = self.guidance_model(
                    est_clean_pos.cpu(), species.cpu(), cell.cpu()
                )[1].to(pos.device)
                norm = torch.linalg.norm(
                    self.target_y - pred_y, dim=1, keepdim=True
                )
            elif self.guidance_type in ("xrd", "nd"):
                pred_y = self.guidance_model(est_clean_pos, species, cell)
                norm = torch.linalg.norm(self.target_y - pred_y)
            elif self.guidance_type in ("exafs", "xanes"):
                ei2, ev2 = periodic_radius_graph(est_clean_pos, self.cutoff, cell)
                ea2 = torch.hstack([ev2, ev2.norm(dim=-1, keepdim=True)])
                pred_y = self.guidance_model(species, ei2, ea2)
                if self.spec_average:
                    pred_y = pred_y.mean(dim=0, keepdim=True)
                norm = torch.linalg.norm(
                    self.target_y - pred_y, dim=1, keepdim=True
                )
            else:
                raise ValueError(f"Unknown guidance type: {self.guidance_type}")
            
            loss = norm.square().mean()
            grad = torch.autograd.grad(loss, est_clean_pos)[0]
        
        return -(self.rho / (norm.sum() + 1e-12)) * grad.detach(), norm.detach()
