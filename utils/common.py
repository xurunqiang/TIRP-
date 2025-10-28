import os
import random
import numpy as np
import torch

def make_dirs(path):
    """创建目录（若不存在）"""
    os.makedirs(path, exist_ok=True)

def set_seed(seed):
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
