"""
@Author: Conghao Wong
@Date: 2024-10-08 19:10:27
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-10 16:53:32
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import ResonanceArgs
from .model import ResonanceModel, ResonanceStructure

# Register new args and models
qpid.register_args(ResonanceArgs, 'Resonance Args')
qpid.register(
    re=[ResonanceStructure, ResonanceModel],
)
