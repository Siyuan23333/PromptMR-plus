"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        # --- DEBUG: check inputs ---
        if (data_range == 0).any():
            logger.warning(f"[SSIMLoss] data_range has zeros: {data_range.flatten().tolist()}")
        if torch.isnan(X).any():
            logger.warning(f"[SSIMLoss] NaN in X (prediction)! shape={X.shape}")
        if torch.isnan(Y).any():
            logger.warning(f"[SSIMLoss] NaN in Y (target)! shape={Y.shape}")
        if torch.isinf(X).any():
            logger.warning(f"[SSIMLoss] Inf in X (prediction)!")
        if torch.isinf(Y).any():
            logger.warning(f"[SSIMLoss] Inf in Y (target)!")

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        # --- DEBUG: check SSIM computation ---
        if torch.isnan(S).any():
            logger.warning(f"[SSIMLoss] NaN in SSIM map! "
                           f"D min={D.min():.4e} D_has_zero={(D==0).any().item()} "
                           f"C1={C1.flatten().tolist()} C2={C2.flatten().tolist()}")

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S
