# Deterministik çalıştırma için seed ayarları

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int, deterministic: bool = False, cuda_deterministic: Optional[bool] = None) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic is None:
        flag = deterministic
    else:
        flag = cuda_deterministic
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(flag)
        torch.backends.cudnn.benchmark = not bool(flag)
