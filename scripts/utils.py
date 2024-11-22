"""
@Author: Conghao Wong
@Date: 2022-11-11 09:28:52
@LastEditors: Conghao Wong
@LastEditTime: 2024-11-05 20:49:53
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np

TK_BORDER_WIDTH = 5
TK_TITLE_STYLE = dict(font=('', 24, 'bold'),
                      height=2)


def get_value(key: str, args: list[str], default=None):
    """
    `key` is started with `--`.
    For example, `--logs`.
    """
    args = np.array(args)
    index = np.where(args == key)[0][0]

    try:
        return str(args[index+1])
    except IndexError:
        return default
