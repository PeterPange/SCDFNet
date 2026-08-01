"""CSPC: Consistency-Saliency Pre-fusion Calibration.

Wraps CMCA (Cross-Modal Consistency Alignment) followed by SCDS
(Saliency-Calibrated Discrepancy Suppression), with a weak residual update
controlled by a learnable scalar alpha. Applied before each fusion stage.
"""

import torch
import torch.nn as nn

from .cmca import CMCA
from .scds import SCDS


class CSPC(nn.Module):
    """Consistency-Saliency Pre-fusion Calibration (CSPC).

    Applied before each of the first two fusion stages. Composed of CMCA
    (Cross-Modal Consistency Alignment) followed by SCDS (Saliency-Calibrated
    Discrepancy Suppression), with a weak residual update controlled by a
    learnable scalar alpha:

        f'_c = CMCA(f)
        f'_s = SCDS(f'_c)
        f_out = f'_c + alpha * (f'_s - f'_c)
    """

    def __init__(self, dim, lambda_init=0.2, alpha_init=0.1):
        super().__init__()
        self.cmca = CMCA(input_dim=dim, lambda_init=lambda_init)
        self.scds = SCDS(dim=dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, fin, fvi):
        fin_cmca, fvi_cmca = self.cmca(fin, fvi)
        fin_scds, fvi_scds = self.scds(fin_cmca, fvi_cmca)
        fin_out = fin_cmca + self.alpha * (fin_scds - fin_cmca)
        fvi_out = fvi_cmca + self.alpha * (fvi_scds - fvi_cmca)
        return fin_out, fvi_out
