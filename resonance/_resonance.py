"""
@Author: Conghao Wong
@Date: 2024-10-10 18:26:32
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-19 20:11:08
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer
from qpid.utils import get_mask


class ResonanceLayer(torch.nn.Module):
    """
    ResonanceLayer
    ---
    Compute resonance features for each neighor, and gather these features
    into the resonance matrix.
    """

    def __init__(self, partitions: int,
                 hidden_units: int,
                 output_units: int,
                 transform_layer: _BaseTransformLayer,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.d_h = hidden_units
        self.d = output_units
        self.T_layer = transform_layer

        # Shapes
        self.Trsteps_en, self.Trchannels_en = self.T_layer.Tshape

        # Trajectory encoding (neighbors)
        self.tre = layers.TrajEncoding(self.T_layer.Oshape[-1], hidden_units,
                                       torch.nn.ReLU,
                                       transform_layer=self.T_layer)

        self.fc1 = layers.Dense(hidden_units*self.Trsteps_en,
                                hidden_units,
                                torch.nn.ReLU)
        self.fc2 = layers.Dense(hidden_units, hidden_units, torch.nn.ReLU)
        self.fc3 = layers.Dense(hidden_units, output_units//2, torch.nn.ReLU)

        # Circle encoding (only for other components)
        self.ce = layers.TrajEncoding(2, output_units//2, torch.nn.ReLU)

    def forward(self, x_ego_2d: torch.Tensor,
                x_nei_2d: torch.Tensor):

        # Move the last point of trajectories to 0
        x_ego_pure = (x_ego_2d - x_ego_2d[..., -1:, :])[..., None, :, :]
        x_nei_pure = x_nei_2d - x_nei_2d[..., -1:, :]

        # Embed trajectories (ego + neighbor) together and then split them
        f_pack = self.tre(torch.concat([x_ego_pure, x_nei_pure], dim=-3))
        f_ego = f_pack[..., :1, :, :]
        f_nei = f_pack[..., 1:, :, :]

        # Compute meta resonance features (for each neighbor)
        # shape of the final output `f_re_meta`: (batch, N, d/2)
        f = f_ego * f_nei   # -> (batch, N, obs, d)
        f = torch.flatten(f, start_dim=-2, end_dim=-1)
        f_re = self.fc3(self.fc2(self.fc1(f)))

        # Compute positional information in a SocialCircle-like way
        # `x_nei_2d` are relative values to target agents' last obs step
        p_nei = x_nei_2d[..., -1, :]

        # Compute distances and angles (for all neighbors)
        f_distance = torch.norm(p_nei, dim=-1)
        f_angle = torch.atan2(p_nei[..., 0], p_nei[..., 1])
        f_angle = f_angle % (2 * np.pi)

        # Partitioning
        partition_indices = f_angle / (2*np.pi/self.partitions)
        partition_indices = partition_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(x_nei_2d, dim=[-1, -2]), torch.int32)
        partition_indices = partition_indices * nei_mask + -1 * (1 - nei_mask)

        positions = []
        re_partitions = []
        for _p in range(self.partitions):
            _mask = (partition_indices == _p).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001

            positions.append([])
            positions[-1].append(torch.sum(f_distance * _mask, dim=-1) / n)
            positions[-1].append(torch.sum(f_angle *
                                           _mask, dim=-1) / n)

            re_partitions.append(
                torch.sum(f_re * _mask[..., None], dim=-2) / n[..., None])

        # Stack all partitions
        positions = [torch.stack(i, dim=-1) for i in positions]
        positions = torch.stack(positions, dim=-2)
        re_partitions = torch.stack(re_partitions, dim=-2)

        # Encode circle components -> (batch, partition, d/2)
        f_pos = self.ce(positions)

        # Concat resonance features -> (batch, partition, d)
        re_matrix = torch.concat([re_partitions, f_pos], dim=-1)

        return re_matrix, f_re
