import torch
import numpy as np
import lightning.pytorch as pl

####################### Dataset #######################

import ase.io
from pathlib import Path
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.transform import Rotation

from glass.nn import periodic_radius_graph
from glass.diffusion import VarianceExplodingDiffuser


class StructureSpecDataset(Dataset):
    def __init__(
        self,
        data_dir,
        cutoff,
        k,
        train_prior=True,
        train_size=0.9,
        scale_y=1.0,
        dup=1,
        guide_type=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cutoff = cutoff
        self.diffuser = VarianceExplodingDiffuser(k=k)
        self.train_prior = train_prior
        self.train_size = train_size
        self.scale_y = scale_y
        self.dup = dup
        self.guide_type = guide_type

        # File names to read
        self.X_fnames = sorted(Path(data_dir).glob("structures/train/*.xyz"))
        if not self.X_fnames:
            self.X_fnames = sorted(Path(data_dir).glob("structures/*.xyz"))
        if not self.X_fnames:
            self.X_fnames = sorted(Path(data_dir).glob("*.xyz"))

        if not self.X_fnames:
            raise FileNotFoundError(
                f"No .xyz files found in '{data_dir}'. "
                f"Searched: structures/train/*.xyz, structures/*.xyz, *.xyz. "
                f"Please ensure your data is in one of these locations."
            )

        atoms_list = [ase.io.read(f) for f in self.X_fnames]

        if guide_type is not None:
            self.y_fnames = sorted(
                Path(data_dir).glob(f"{self.guide_type}/train/*.txt")
            )

            # Make sure the structure files match the spectra files
            for i, _ in enumerate(self.X_fnames):
                assert self.X_fnames[i].stem == self.y_fnames[i].stem

            spec_list = [np.genfromtxt(f) for f in self.y_fnames]

        # Wrap atoms outside the periodic boundaries
        for atoms in atoms_list:
            atoms.wrap()

        # Onehot encoder for atom type
        unique_numbers = np.concatenate(
            [np.unique(atoms.numbers) for atoms in atoms_list]
        )
        self.atom_encoder = OneHotEncoder(sparse_output=False)
        self.atom_encoder.fit(unique_numbers.reshape(-1, 1))

        # Store list of PyG Data objects as the dataset
        self.dataset = []
        if guide_type in ["xanes", "exafs"]:
            for atoms, spec in zip(atoms_list, spec_list):
                z = self.atom_encoder.transform(atoms.numbers.reshape(-1, 1))
                data = Data(
                    z=torch.tensor(z, dtype=torch.float),
                    pos=torch.tensor(atoms.positions, dtype=torch.float),
                    y=torch.tensor(spec * scale_y, dtype=torch.float),
                    train_mask=torch.rand(len(z)) < train_size,
                    cell=np.array(atoms.cell),
                )
                # Duplicate data for longer epochs
                for _ in range(dup):
                    self.dataset.append(data.clone())

        elif guide_type is None:
            for atoms in atoms_list:
                z = self.atom_encoder.transform(atoms.numbers.reshape(-1, 1))
                data = Data(
                    z=torch.tensor(z, dtype=torch.float),
                    pos=torch.tensor(atoms.positions, dtype=torch.float),
                    y=torch.tensor(
                        scale_y, dtype=torch.float
                    ),  # when guide_type is None, y no longer useful
                    train_mask=torch.rand(len(z)) < train_size,
                    cell=np.array(atoms.cell),
                )
                for _ in range(dup):
                    self.dataset.append(data.clone())

        else:
            raise ValueError(f"[✗] Unrecognized guide_type: {guide_type}")

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx].clone()
        if self.train_prior:
            data = self._diffuse_pos(data)
        data = self._atomic_graph(data, cutoff=self.cutoff)
        data = self._random_rotate(data)
        return data

    def _diffuse_pos(self, data):
        data.t = (
            torch.rand(1)
            .clip(self.diffuser.t_min, self.diffuser.t_max)
            .expand(data.pos.size(0), 1)
        )
        data.pos, data.eps_r = self.diffuser.forward_noise(data.pos, data.t)
        data.sigma_r = self.diffuser.sigma(data.t)
        return data

    def _atomic_graph(self, data, cutoff):
        cell = torch.tensor(data.cell, dtype=torch.float)
        data.edge_index, edge_vec = periodic_radius_graph(data.pos, cutoff, cell=cell)
        data.edge_len = edge_vec.norm(dim=-1, keepdim=True)
        data.edge_attr = torch.hstack([edge_vec, data.edge_len])
        return data

    def _random_rotate(self, data):
        R = torch.tensor(
            Rotation.random().as_matrix(), dtype=torch.float, device=data.pos.device
        )
        data.pos @= R
        data.edge_attr[:, :3] @= R
        if self.train_prior:
            data.eps_r @= R
        return data


####################### LightningDataModule #######################

from torch_geometric.loader import DataLoader


class StructureSpecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        cutoff,
        k,
        train_prior=True,
        train_size=0.9,
        scale_y=1.0,
        dup=1,
        guide_type=None,
        batch_size=8,
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.cutoff = cutoff
        self.k = k
        self.train_prior = train_prior
        self.train_size = train_size
        self.scale_y = scale_y
        self.dup = dup
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.guide_type = guide_type

    def prepare_data(self):
        # Download, IO, etc. Useful with shared filesystems
        # Only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # Make assignments here (val/train/test split)
        # Called on every process in DDP
        self.train_set = StructureSpecDataset(
            self.data_dir,
            self.cutoff,
            self.k,
            self.train_prior,
            self.train_size,
            self.scale_y,
            self.dup,
            self.guide_type,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage=None):
        # Clean up state after the trainer stops, delete files...
        # Called on every process in DDP
        pass
