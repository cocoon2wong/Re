"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-15 11:05:49
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import qpid
from playground import PlaygroundArgs
from qpid.mods.vis import VisArgs
from resonance import ResonanceArgs

TARGET_FILE = './README.md'


if __name__ == '__main__':
    qpid.register_args(ResonanceArgs, 'Re Args')
    qpid.register_args(PlaygroundArgs, 'Playground Args')
    qpid.register_args(VisArgs, 'Visualization Args')
    qpid.help.update_readme(qpid.get_all_args_docs(html=True, print_docs=False),
                            TARGET_FILE)
