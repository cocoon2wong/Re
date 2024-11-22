"""
@Author: Conghao Wong
@Date: 2024-10-08 19:18:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-19 20:20:02
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure
from qpid.utils import INIT_POSITION

from .__args import ResonanceArgs
from .__layers import SocialCircleLayer
from ._reBias import ReBiasLayer
from ._resonance import ResonanceLayer
from ._selfBias import SelfBiasLayer
from .linearDiffEncoding import LinearDiffEncoding


class ResonanceModel(Model):
    """
    *Re* Model
    ---
    The *Resonance* trajectory prediction model, short for *Re*.

    Main contributions:
    - The ``vibration-like'' prediction strategy that divides pedestrian
        trajectory prediction into the direct superposition of multiple
        vibration portions, i.e., trajectory biases, to better simulate their
        intuitive behaviors, including the linear base, the self-bias, and the
        resonance-bias;
    - The ``resonance-like'' representation of social interactions when
        forecasting trajectories, which regards that social interactions are
        associated with trajectory spectrums of interaction participators and
        their similarities.
    """

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.as_final_stage_model = True

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Set model inputs
        # Types of agents are only used in complex scenes
        # For other datasets, keep it disabled (through the arg)
        if not self.re_args.encode_agent_types:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ)
        else:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ,
                            INPUT_TYPES.AGENT_TYPES)

        # Layers
        # Transform layers (ego)
        tlayer, itlayer = layers.get_transform_layers(self.re_args.T)
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((len(self.output_pred_steps), self.dim))

        # Transform layers (neighbors)
        trlayer, itrlayer = layers.get_transform_layers(self.re_args.Tr)
        self.tr1 = trlayer((self.args.obs_frames, self.dim))
        self.itr1 = itrlayer((self.args.pred_frames, self.dim))

        # Linear Difference Encoding Layer
        self.linear = LinearDiffEncoding(obs_frames=self.args.obs_frames,
                                         pred_frames=self.args.pred_frames,
                                         output_units=self.d//2,
                                         transform_layer=self.t1,
                                         encode_agent_types=self.re_args.encode_agent_types)

        # Self-Bias Layer
        if self.re_args.learn_self_bias:
            self.b1 = SelfBiasLayer(self.args,
                                    output_units=self.d,
                                    noise_units=self.d//2,
                                    transform_layer=self.t1,
                                    itransform_layer=self.it1)

        if not self.re_args.learn_re_bias:
            return

        # Layer to compute the resonance matrix (or SocialCircle)
        if not self.re_args.use_original_socialcircle:
            self.rc = ResonanceLayer(partitions=self.re_args.partitions,
                                     hidden_units=self.d,
                                     output_units=self.d,
                                     transform_layer=self.tr1)
        else:
            self.rc = SocialCircleLayer(partitions=self.re_args.partitions,
                                        output_units=self.d)

        # Resonance Bias Layer
        self.b2 = ReBiasLayer(self.args,
                              output_units=self.d,
                              noise_units=self.d//2,
                              ego_feature_dim=self.d,
                              re_feature_dim=self.d//2,
                              T_nei_obs=self.tr1,
                              iT_nei_pred=self.itr1)

    def forward(self, inputs: list[torch.Tensor], training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, dim)
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # (batch, N, obs, dim)
        if self.re_args.no_interaction:
            x_nei = self.create_empty_neighbors(x_ego)
        else:
            x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Get types of all agents (if needed)
        if self.re_args.encode_agent_types:
            agent_types = self.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
        else:
            agent_types = None

        # Encode features of ego trajectories (diff encoding)
        f_diff, linear_fit, linear_base = self.linear(x_ego, agent_types)

        # Predict the self-bias trajectory
        if self.re_args.learn_self_bias:
            self_bias = self.b1(linear_fit, f_diff,
                                self.output_pred_steps, training)
        else:
            self_bias = 0

        # Predict the re-bias trajectory
        if self.re_args.learn_re_bias:
            # Compute and encode the Resonance feature to each ego agent
            # `re_matrix`: Resonance Matrix
            # `f_re`: Resonance feature
            re_matrix, f_re = self.rc(self.picker.get_center(x_ego)[..., :2],
                                      self.picker.get_center(x_nei)[..., :2])

            # Compute the resonance-bias trajectory
            re_bias = self.b2(x_ego - linear_fit,
                              f_diff, re_matrix, training)
        else:
            re_bias = 0

        # -----------------------
        # # # # The following lines are used to draw visualized figures in our paper
        # from scripts.draw_neighbor_contributions import draw, draw_spectrums, draw_pca
        # from scripts.draw_partitions import draw_partitions
        # draw(self, self.get_top_manager().args.force_clip, x_ego, x_nei, f_re)
        # draw_pca(x_nei, f_re)
        # draw_spectrums(x_nei, self.tr1)

        # w = self.b2.concat_fc.linear.weight
        # d = self.d//2

        # _f_re = f_re[..., :d]
        # _f_pos = f_re[..., d:2*d]
        # w_re = w[..., d:2*d]
        # w_pos = w[..., 2*d:3*d]

        # draw_partitions(_f_re[:1] @ w_re.T, 're_pool',
        #                 color_high=[0xf9, 0xcf, 0x62],
        #                 color_low=[0x74, 0x8b, 0xe2],
        #                 max_width=0.3, min_width=0.2)
        # draw_partitions(_f_pos[:1] @ w_pos.T, 'pos_pool',
        #                 color_high=[0xf9, 0x5c, 0x77],
        #                 color_low=[0x74, 0x8b, 0xe2],
        #                 max_width=-0.3, min_width=-0.2)
        # # # # Vis codes end here
        # -----------------------

        # Add all biases to the base trajectory to compute the final prediction
        if not self.re_args.disable_linear_base:
            y = linear_base[..., None, :, :]
        else:
            y = 0

        if training or not self.re_args.no_self_bias:
            y = y + self_bias

        if training or not self.re_args.no_re_bias:
            y = y + re_bias

        return y

    def create_empty_neighbors(self, x_ego: torch.Tensor):
        """
        Create the neighbor trajectory matrix that only contains the ego agent.
        """
        empty = INIT_POSITION * torch.ones([x_ego.shape[0],
                                            self.args.max_agents - 1,
                                            x_ego.shape[-2],
                                            x_ego.shape[-1]]).to(x_ego.device)
        return torch.concat([x_ego[..., None, :, :], empty], dim=-3)


class ResonanceStructure(Structure):
    MODEL_TYPE = ResonanceModel
