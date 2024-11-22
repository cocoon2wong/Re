"""
@Author: Conghao Wong
@Date: 2024-10-31 20:03:29
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-11 15:18:39
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from qpid.model import Model
from qpid.mods import vis
from qpid.utils import dir_check, get_mask

COLOR_HIGH = np.array([0xe7, 0xe7, 0x71], np.float32)
# COLOR_MID = np.array([0x1e, 0x9e, 0xee], np.float32)
COLOR_LOW = np.array([0xf8, 0x77, 0x61], np.float32)

ROOT_DIR = './temp_files/re'

v = None


def draw(model: Model, clip: str,
         obs: torch.Tensor,
         nei: torch.Tensor,
         f_re_meta: torch.Tensor,
         w_max=0.5,
         w_min=0.4):

    global v
    if v is None:
        v = vis.Visualization(manager=model.manager,
                              dataset=model.args.dataset,
                              clip=clip)

    nei_count = int(torch.sum(get_mask(torch.sum(nei[0], dim=[-1, -2]))))
    _nei = nei[0, :nei_count]
    _obs = obs[0]

    r_max = w_max
    r_min = w_min
    text_delta = r_min

    # Open a new canvas
    plt.close('nei_contribution')
    plt.figure('nei_contribution')
    v._visualization_plt(None, obs=_obs, neighbor=_nei)

    # Compute radiuses
    r_re_real = torch.sum(f_re_meta ** 1, dim=-1)[0, :nei_count]
    r = (r_re_real/torch.max(r_re_real))

    # Draw each neighbor
    for index, (_nei_p, _r) in enumerate(zip(_nei[..., -1, :], r)):
        _radius = (r_min + (r_max - r_min) * _r).numpy()
        _color = COLOR_LOW + (COLOR_HIGH - COLOR_LOW) * _r.numpy()
        _pos = (float(_nei_p[0]), float(_nei_p[1]))

        _circle = plt.Circle(_pos, _radius,
                             fill=True, color=list(_color/255),
                             alpha=0.9)
        plt.gca().add_artist(_circle)

        plt.text(_pos[0] + text_delta, _pos[1] + text_delta, str(index),
                 color='white',
                 fontsize=20,
                 bbox=dict(boxstyle='round', alpha=0.5))

    plt.show()


def draw_pca(nei: torch.Tensor,
             f_re_meta: torch.Tensor,
             w_max=0.1,
             w_min=0.06):

    nei_count = int(torch.sum(get_mask(torch.sum(nei[0], dim=[-1, -2]))))
    a = f_re_meta[0, :nei_count]
    u, s, v = torch.pca_lowrank(a, q=2)
    _p = torch.matmul(a, v[:, :2]).numpy()

    _p = _p / np.array([np.max(np.abs(_p[..., 0])),
                        np.max(np.abs(_p[..., 1]))])

    r_max = w_max
    r_min = w_min
    text_delta = r_min

    plt.close('nei_contribution_pca')
    plt.figure('nei_contribution_pca')

    # Compute radiuses
    r_re_real = torch.sum(f_re_meta ** 1, dim=-1)[0, :nei_count]
    r = (r_re_real/torch.max(r_re_real))

    for index, (_nei_p, _r) in enumerate(zip(_p, r)):
        _radius = (r_min + (r_max - r_min) * _r).numpy()
        _color = COLOR_LOW + (COLOR_HIGH - COLOR_LOW) * _r.numpy()
        _pos = (float(_nei_p[0]), float(_nei_p[1]))

        plt.plot(*_pos, 'o')

        _circle = plt.Circle(_pos, _radius,
                             fill=True, color=list(_color/255),
                             alpha=0.9)
        plt.gca().add_artist(_circle)

        plt.text(_nei_p[0] + text_delta, _nei_p[1] + text_delta, str(index),
                 color='white',
                 fontsize=20,
                 bbox=dict(boxstyle='round', alpha=0.5))

        plt.axis('equal')

    # Plot corner points to resize the canvas
    for x in [-1.5, 1.5]:
        for y in [-1.5, 1.5]:
            plt.plot(x, y)

    plt.show()


def draw_spectrums(nei: torch.Tensor,
                   Tlayer: torch.nn.Module,
                   length_gain=50):

    nei_count = int(torch.sum(get_mask(torch.sum(nei[0], dim=[-1, -2]))))
    nei_spec = Tlayer(nei[0, :nei_count] - nei[0, :nei_count, -1:, :])

    # Compute new shape (after resize)
    shape = nei_spec[0].shape
    new_shape = (shape[0] * length_gain,
                 shape[1] * length_gain)

    # Batch normalize -> (0, 1)
    nei_spec = (nei_spec - nei_spec.min()) / (nei_spec.max() - nei_spec.min())

    r = dir_check(ROOT_DIR)
    for index, _spec in enumerate(nei_spec):
        p = os.path.join(r, f'{index}.png')

        _spec = (255 * _spec).numpy().astype(np.uint8)
        _spec = cv2.resize(_spec, new_shape, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(p, _spec)
