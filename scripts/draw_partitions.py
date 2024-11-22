"""
@Author: Conghao Wong
@Date: 2024-11-01 15:51:00
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-06 10:08:07
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from qpid.utils import dir_check

ROOT_DIR = './temp_files/re'

COLOR_HIGH = [250, 100, 100]
COLOR_LOW = [60, 100, 220]

MAX_WIDTH = 0.3
MIN_WIDTH = 0.1


def cal_color(weights: list[float], color_high: list[int], color_low: list[int]):
    w: np.ndarray = np.array(weights)
    w = w - w.min()
    w /= (w.max() - w.min())
    _color_high = np.array(color_high)
    _color_low = np.array(color_low)
    color = _color_low + w[:, np.newaxis] * (_color_high - _color_low)
    return [(i.tolist()) for i in color/255]


def cal_radius(weights: list[float],
               max_value=MAX_WIDTH,
               min_value=MIN_WIDTH) -> np.ndarray:
    w: np.ndarray = np.array(weights)
    return (max_value - min_value) * w/w.max() + min_value


def draw_partitions(f_re: torch.Tensor, file_name: str,
                    color_high: list[int] = COLOR_HIGH,
                    color_low: list[int] = COLOR_LOW,
                    max_width: float = MAX_WIDTH,
                    min_width: float = MIN_WIDTH):

    p = torch.sum(f_re[0] ** 2, dim=-1).numpy()

    fig = plt.figure()
    colors = cal_color(p, color_high, color_low)
    radius = cal_radius(p, max_width, min_width)

    if max_width > 0 and min_width > 0:
        # Draw the big circle
        plt.pie(x=[1 for _ in p],
                radius=1,
                colors=colors)

        # Draw the center circle
        for index, r in enumerate(radius):
            _colors: list = [(0, 0, 0, 0) for _ in p]
            _colors[index] = fig.get_facecolor()
            plt.pie(x=[1 for _ in p],
                    radius=1.0-r,
                    colors=_colors)

    elif max_width < 0 and min_width < 0:
        # Draw the big circle
        for index, r in enumerate(radius):
            _colors: list = [(0, 0, 0, 0) for _ in p]
            _colors[index] = colors[index]
            plt.pie(x=[1 for _ in p],
                    radius=1.0-r,
                    colors=_colors)

        # Draw the center circle
        plt.pie(x=[1 for _ in p],
                radius=1,
                colors=[fig.get_facecolor() for _ in p])

    r = dir_check(ROOT_DIR)
    plt.savefig(f := (os.path.join(r, f'{file_name}.png')))
    plt.close()

    # Save as a png image
    fig_saved: np.ndarray = cv2.imread(f)
    alpha_channel = 255 * (np.min(fig_saved[..., :3], axis=-1) != 255)
    fig_png = np.concatenate(
        [fig_saved, alpha_channel[..., np.newaxis]], axis=-1)

    # Cut the image
    areas = fig_png[..., -1] == 255
    x_value = np.sum(areas, axis=0)
    x_index_all = np.where(x_value)[0]
    y_value = np.sum(areas, axis=1)
    y_index_all = np.where(y_value)[0]

    cv2.imwrite(os.path.join(r, f'{file_name}_cut.png'),
                fig_png[y_index_all[0]:y_index_all[-1],
                        x_index_all[0]:x_index_all[-1]])
