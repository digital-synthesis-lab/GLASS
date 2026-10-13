import torch
import numpy as np

# Typing
from torch import Tensor
from typing import List, Tuple, Optional

####################### Model #######################

from torch import nn
from glass.nn import MLP


class Encoder(nn.Module):
    def __init__(
        self, init_node_dim: int, init_edge_dim: int, node_dim: int, edge_dim: int
    ) -> None:
        super().__init__()
        self.init_node_dim = init_node_dim
        self.init_edge_dim = init_edge_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.embed_node = nn.Sequential(
            MLP([init_node_dim, node_dim, node_dim], act=nn.SiLU()),
            nn.LayerNorm(node_dim),
        )
        self.embed_edge = nn.Sequential(
            MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU()),
            nn.LayerNorm(edge_dim),
        )

    def forward(self, x: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        h_node = self.embed_node(x)
        h_edge = self.embed_edge(edge_attr)
        return h_node, h_edge


class Decoder(nn.Module):
    def __init__(self, node_dim: int, out_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())

    def forward(self, h_node: Tensor) -> Tensor:
        out = self.decoder(h_node)
        # return torch.nn.functional.softplus(out) # softplus enforces positive outputs
        return out  # for exafs


class MeshGraphNets(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h_node, h_edge = self.encoder(x, edge_attr)
        h_node, h_edge = self.processor(h_node, edge_index, h_edge)
        return self.decoder(h_node)


####################### LightningModule #######################

import lightning as L
from glass.nn.mgn import Processor


class LitSpecNet(L.LightningModule):
    def __init__(
        self,
        num_species: int,
        num_convs: int,
        dim: int,
        out_dim: int,
        ema_decay: float = 0.999,
        learn_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Core model
        self.model = MeshGraphNets(
            encoder=Encoder(num_species, 3 + 1, dim, dim),
            processor=Processor(num_convs, dim, dim),
            decoder=Decoder(dim, out_dim),
        )

        # EMA model
        ema_avg = (
            lambda avg_params, params, num_avg: ema_decay * avg_params
            + (1 - ema_decay) * params
        )
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # Training parameters
        self.learn_rate = learn_rate

    def training_step(self, batch, batch_idx):

        pred_y = self.model(batch.z, batch.edge_index, batch.edge_attr)
        mask = batch.train_mask

        train_loss = torch.nn.functional.mse_loss(pred_y[mask], batch.y[mask])
        self.log(
            "train_loss",
            train_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        valid_loss = torch.nn.functional.mse_loss(pred_y[~mask], batch.y[~mask])
        self.log(
            "valid_loss",
            valid_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            "hp_metric",
            valid_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.num_graphs,
        )
        return train_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema_model.update_parameters(self.model)
