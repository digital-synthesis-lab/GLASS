import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union
import ase
from sklearn.preprocessing import OneHotEncoder


def initialize_atoms(init_atoms):
    node_encoder = OneHotEncoder(sparse_output=False)
    x = node_encoder.fit_transform(init_atoms.numbers.reshape(-1, 1))
    init_species = torch.tensor(x, dtype=torch.float)
    init_pos = torch.tensor(init_atoms.positions, dtype=torch.float)
    init_cell = torch.tensor(init_atoms.cell.tolist(), dtype=torch.float)
    Z_list = node_encoder.categories_[0].tolist()
    return Z_list, init_species, init_pos, init_cell


def atoms_to_device(
    atoms: ase.Atoms,
    device: Union[str, torch.device],
) -> Tuple[Tensor, Tensor, Tensor]:
    device = torch.device(device)
    _, species, pos, cell = initialize_atoms(atoms)
    return (
        species.to(device),
        pos.to(device=device, dtype=torch.float32),
        cell.to(device=device, dtype=torch.float32),
    )


def compute_prior_score(
    species: Tensor,
    pos: Tensor,
    cell: Tensor,
    t: Tensor,
    cutoff: float,
    score_net,
    diffuser,
) -> Tensor:
    from glass.nn import periodic_radius_graph

    edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
    edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
    return score_net.ema_model(species, edge_index, edge_attr, t, diffuser.sigma(t))


def compute_target_from_reference(
    ref_atoms: ase.Atoms,
    guidance_model,
    guidance_type: str,
    cutoff: float,
    device: Union[str, torch.device],
) -> Tensor:
    from glass.nn import periodic_radius_graph

    device = torch.device(device)
    _, ref_species, ref_pos, ref_cell = initialize_atoms(ref_atoms)

    if guidance_type in ("pdf", "adf"):
        return guidance_model(
            ref_pos.cpu(), ref_species.cpu(), ref_cell.cpu()
        )[1].to(device)

    elif guidance_type in ("xrd", "nd"):
        return guidance_model(
            ref_pos.to(device), ref_species.to(device), ref_cell.to(device)
        )

    elif guidance_type in ("exafs", "xanes"):
        ei_r, ev_r = periodic_radius_graph(
            ref_pos.to(device), cutoff, ref_cell.to(device)
        )
        ea_r = torch.hstack([ev_r, ev_r.norm(dim=-1, keepdim=True)])
        with torch.no_grad():
            return guidance_model(ref_species.to(device), ei_r, ea_r)

    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
