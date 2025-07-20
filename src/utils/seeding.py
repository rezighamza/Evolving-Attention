# src/utils/seeding.py
import torch
import numpy as np
import random

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The two lines below can slow down training but are needed for full determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False