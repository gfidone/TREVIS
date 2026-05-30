import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
from typing import List, Tuple

class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (S, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, S, D)

        self.register_buffer("pe", pe)

    def forward(self, x, start_pos=0):
        # x: (B, S, D)
        S = x.size(1)
        return self.pe[:, start_pos:start_pos + S, :] 

class AbsoluteTreeEncoder(nn.Module):
    def __init__(self,
                 num_scales
                ):
        super().__init__()
        self.num_scales = num_scales
        self.p_raw = nn.Parameter(torch.linspace(-1.0, 1.0, self.num_scales))  

    def forward(self, abs_encs):

        B = abs_encs.shape[0]
        S = abs_encs.shape[1]
        pos_dim = abs_encs.shape[2]
        k = pos_dim // 2
        device = abs_encs.device
        
        pos_chunks = abs_encs.view(B, S, k, 2)
    
        p = torch.tanh(self.p_raw)  
        m = self.num_scales
    
        depths = torch.arange(k, device=device).float()  # [k]
        p_powers = p.unsqueeze(1) ** depths.unsqueeze(0)  # [m, k]
        p_powers = p_powers.view(1, 1, m, k, 1)
        pos_expanded = pos_chunks.unsqueeze(2).expand(-1, -1, m, -1, -1)
        scaled = pos_expanded * p_powers
        scaled = scaled.reshape(B, S, m * k * 2)

        return scaled

class RelativeTreeEncoder(nn.Module):
    def __init__(self, alpha=1.0, special_distance=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))

    def forward(self, dists):  
        return -self.alpha * dists

        


