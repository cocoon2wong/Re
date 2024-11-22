"""
@Author: Conghao Wong
@Date: 2024-10-09 20:26:00
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-19 20:15:22
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.model import layers, transformer
from qpid.model.layers.interpolation import (LinearPositionInterpolation,
                                             LinearSpeedInterpolation)
from qpid.model.layers.transfroms import _BaseTransformLayer

from .__args import ResonanceArgs


class SelfBiasLayer(torch.nn.Module):
    """
    Self-Bias Layer
    ---
    Self-bias is used to model diversity and intention changes of the ego agent
    itself as an additional vibration onto the linear base.
    """

    def __init__(self, Args: Args,
                 output_units: int,
                 noise_units: int,
                 transform_layer: _BaseTransformLayer,
                 itransform_layer: _BaseTransformLayer,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.args = Args
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Layers
        # Transform layers (ego)
        self.T_layer = transform_layer
        self.iT_layer = itransform_layer

        self.d = output_units
        self.d_z = noise_units
        self.d_id = self.args.noise_depth

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.T_layer.Tshape
        self.Tsteps_de, self.Tchannels_de = self.iT_layer.Tshape

        # Noise encoding
        self.ie = layers.TrajEncoding(
            self.d_id, self.d - self.d_z, torch.nn.Tanh)

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_de,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        # Decoder layers
        # See "MSN: Multi-Style Network for Trajectory Prediction" for detail
        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(self.d, self.re_args.Kc, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)
        self.decoder_fc1 = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.d,
                                        self.Tsteps_de * self.Tchannels_de)

        # Interpolation layer
        match self.re_args.interp:
            case 'linear':
                i_layer_type = LinearPositionInterpolation
            case 'speed':
                i_layer_type = LinearSpeedInterpolation
            case _:
                raise ValueError

        self.interp_layer = i_layer_type()

    def forward(self, linear_fit: torch.Tensor,
                f_diff: torch.Tensor,
                waypoints_indices: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        traj_targets = self.T_layer(linear_fit)

        # First predict the overall waypoint-bias (on several waypoints)
        for _ in range(repeats):
            # Assign random noise and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_diff.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(linear_fit.device))

            # (batch, steps, 2*d)
            f_final = torch.concat([f_diff, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Multiple generations -> (batch, Kc, d)
            adj = self.ms_fc(f_final)               # (batch, steps, Kc)
            adj = torch.transpose(adj, -1, -2)
            f_multi = self.ms_conv(f_tran, adj)     # (batch, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = torch.reshape(y, list(y.shape[:-1]) +
                              [self.Tsteps_de, self.Tchannels_de])

            y = self.iT_layer(y)
            all_predictions.append(y)

        # Stack random output -> (batch, K, n_key, dim)
        waypoint_bias = torch.concat(all_predictions, dim=-3)

        # Interpolating waypoints -> (batch, K, pred, dim)
        self_bias = self.interp(index=waypoints_indices,
                                value=waypoint_bias,
                                obs_traj=linear_fit)
        return self_bias

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:
        """
        Interpolate trajectories according to the predicted keypoints.

        :param index: Indices of predictions, which only includs future points.
        :param value: Predicted future keypoints. It has the same length as the \
            above `index`.
        :param obs_traj: Observed trajectories.
        """
        _i = torch.concat([torch.tensor([-1]).to(index.device), index])
        _obs = torch.repeat_interleave(obs_traj[..., None, -1:, :],
                                       repeats=value.shape[-3],
                                       dim=-3) \
            if value.ndim != obs_traj.ndim else obs_traj[..., -1:, :]
        _v = torch.concat([_obs, value], dim=-2)

        if self.re_args.interp == 'linear':
            return self.interp_layer(_i, _v)
        elif self.re_args.interp == 'speed':
            v0 = obs_traj[..., -1:, :] - obs_traj[..., -2:-1, :]
            v0 = v0[..., None, :, :] if v0.ndim != value.ndim else v0
            return self.interp_layer(_i, _v, init_speed=v0)
        else:
            raise ValueError
