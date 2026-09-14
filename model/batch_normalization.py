import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        if training:
            ub = np.mean(x, axis=0)
            sig = np.mean((x - ub) ** 2)
            y = ((x - ub) / math.sqrt(sig + eps)) * gamma + beta
            running_mean = np.dot((1 - momentum), running_mean) + momentum * ub
            running_var = np.dot((1 - momentum), running_var) + momentum * sig
            
        else:
            y = ((x - running_mean) / np.sqrt(running_var + eps)) * gamma + beta
        
        return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4))
