"""CMCA: Cross-Modal Consistency Alignment.

Performs a soft consistency pre-alignment of the two modalities without
cosine similarity or hard thresholds. Part of CSPC.
"""

import torch
import torch.nn as nn
from einops import rearrange


class CMCA(nn.Module):
    """Cross-Modal Consistency Alignment (CMCA).

    Performs a soft consistency pre-alignment of the two modalities without
    cosine similarity or hard thresholds. Returns aligned (fin_out, fvi_out).
    """

    def __init__(self, input_dim, lambda_init=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 1)
        )
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, fin, fvi):
        b, c, h, w = fin.shape

        fin_seq = rearrange(fin, 'b c h w -> b (h w) c')
        fvi_seq = rearrange(fvi, 'b c h w -> b (h w) c')

        diff_seq = torch.abs(fin_seq - fvi_seq)

        refine_input = torch.cat([fin_seq, fvi_seq, diff_seq], dim=-1)  # [B, HW, 3C]

        msc = self.sigmoid(self.mlp(refine_input))  # [B, HW, 1]

        pin = fin_seq * (1.0 - msc) + fvi_seq * msc
        pvi = fvi_seq * (1.0 - msc) + fin_seq * msc

        kin = fin_seq - pin
        kvi = fvi_seq - pvi

        afin = fin_seq - self.lambda_param * kin
        afvi = fvi_seq - self.lambda_param * kvi

        fin_out = rearrange(afin, 'b (h w) c -> b c h w', h=h, w=w)
        fvi_out = rearrange(afvi, 'b (h w) c -> b c h w', h=h, w=w)

        return fin_out, fvi_out
