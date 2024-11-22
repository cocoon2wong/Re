"""
@Author: Conghao Wong
@Date: 2024-10-10 17:24:30
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-19 20:19:20
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer
from qpid.utils import MAX_TYPE_NAME_LEN


class LinearDiffEncoding(torch.nn.Module):
    """
    Linear Difference Encoding Layer
    ---
    It is used to encode the difference between the observed trajectory and
    the corresponding linear (least square) trajectory of the ego agent.
    """

    def __init__(self, obs_frames: int,
                 pred_frames: int,
                 output_units: int,
                 transform_layer: _BaseTransformLayer,
                 encode_agent_types: bool | int = False,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.d = output_units
        self.T_layer = transform_layer

        self.obs_frames = obs_frames
        self.pred_frames = pred_frames

        self.encode_agent_types = encode_agent_types

        # Linear prediction layer
        self.linear = layers.LinearLayerND(self.obs_frames,
                                           self.pred_frames,
                                           return_full_trajectory=True)

        # Trajectory encoding (ego)
        self.te = layers.TrajEncoding(self.T_layer.Oshape[-1], self.d,
                                      torch.nn.ReLU,
                                      transform_layer=self.T_layer)

        # Linear trajectory encoding
        self.le = layers.TrajEncoding(self.T_layer.Oshape[-1], self.d,
                                      torch.nn.ReLU,
                                      transform_layer=self.T_layer)

        # Bilinear structure (outer product + pooling + fc)
        # See "Another vertical view: A hierarchical network for heterogeneous
        # trajectory prediction via spectrums."
        self.outer = layers.OuterLayer(self.d, self.d)
        self.pooling = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = layers.Dense((self.d//2)**2, self.d, torch.nn.Tanh)
        self.outer_fc_linear = layers.Dense((self.d//2)**2, self.d,
                                            torch.nn.Tanh)

        if self.encode_agent_types:
            self.type_encoder = layers.Dense(MAX_TYPE_NAME_LEN, output_units,
                                             torch.nn.Tanh)

    def forward(self, x_ego: torch.Tensor,
                agent_types: torch.Tensor | None = None,
                *args, **kwargs):

        # Compute the linear trajectory
        traj_linear = self.linear(x_ego)      # (batch, obs+pred, dim)

        # Move the linear trajectory to make it intersect with the obs trajectory
        # at the current observation moment (by moving it to (0. 0)).
        _t = self.obs_frames
        traj_linear = traj_linear - traj_linear[..., _t-1:_t, :]
        linear_fit = traj_linear[..., :_t, :]
        linear_base = traj_linear[..., _t:, :]

        # Trajectory embedding and encoding
        f = self.te(x_ego)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_ego = self.outer_fc(f)       # (batch, steps, d/2)

        # Linear trajectory embedding and encoding
        f_l = self.le(linear_fit)
        f_l = self.outer(f_l, f_l)
        f_l = self.pooling(f_l)
        f_l = self.flatten(f_l)
        f_ego_linear = self.outer_fc_linear(f_l)       # (batch, steps, d/2)

        f_diff = f_ego - f_ego_linear    # ranged from (-2, 2)
        f_diff = f_diff / 2           # ranged from (-1 ,1)

        if self.encode_agent_types and (agent_types is not None):
            f_type = self.type_encoder(
                agent_types)[..., None, :]    # (batch, 1, d)
            f_diff = f_diff + f_type

        return f_diff, linear_fit, linear_base
