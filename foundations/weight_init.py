import torch
import torch.nn as nn
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return (torch.randn(fan_out, fan_in) * std).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        return (torch.randn(fan_out, fan_in) * std).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        if init_type == 'xavier':
            std = math.sqrt(2.0 / (input_dim + hidden_dim))
            w = torch.randn(input_dim, hidden_dim) * std
        elif init_type == 'kaiming':
            std = math.sqrt(2.0 / (input_dim))
            w = torch.randn(input_dim, hidden_dim) * std
        else:
            w = torch.randn(input_dim, hidden_dim)
        weights.append(w)

        for _ in range(num_layers-1):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (input_dim + hidden_dim))
                w = torch.randn(hidden_dim, hidden_dim) * std
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / (input_dim))
                w = torch.randn(hidden_dim, hidden_dim) * std
            else:
                w = torch.randn(hidden_dim, hidden_dim)
            weights.append(w)
        
        x = torch.randn(1, input_dim)
        stds = []

        for i in range(num_layers):
            x = x @ weights[i].T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
        
        return stds

