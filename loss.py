import torch
import torch.nn as nn
import torch.nn.functional as F


class ConjLoss(nn.Module):
    def __init__(self,
                 lambda_phys=0.25,
                 lambda_phys_fg=1.0,
                 lambda_a=0.2,
                 lambda_C=0.5,
                 lambda_phi=4.0,
                 lambda_phi_all=0.5,
                 eps=1e-6):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.lambda_phys_fg = lambda_phys_fg
        self.lambda_a = lambda_a
        self.lambda_C = lambda_C
        self.lambda_phi = lambda_phi
        self.lambda_phi_all = lambda_phi_all
        self.eps = eps

    def _masked_mean(self, value, mask=None):
        if mask is None:
            return value.mean()
        return (value * mask).sum() / mask.sum().clamp(min=self.eps)

    def forward(self, a, C_real, C_imag, fringe, theta, phi_gt, mask,
                a_gt, C_gt_real, C_gt_imag):
        I_recon = a + 2.0 * C_real * torch.cos(theta) - 2.0 * C_imag * torch.sin(theta)

        L_phys = F.mse_loss(I_recon, fringe)
        L_phys_fg = self._masked_mean((I_recon - fringe) ** 2, mask)
        L_a = F.mse_loss(a, a_gt)
        L_C = F.mse_loss(C_real, C_gt_real) + F.mse_loss(C_imag, C_gt_imag)

        phi_pred = torch.atan2(C_imag, C_real)
        phi_err = 1.0 - torch.cos(phi_pred - phi_gt)
        L_phi = self._masked_mean(phi_err, mask)
        L_phi_all = phi_err.mean()

        total = (
            self.lambda_phys * L_phys
            + self.lambda_phys_fg * L_phys_fg
            + self.lambda_a * L_a
            + self.lambda_C * L_C
            + self.lambda_phi * L_phi
            + self.lambda_phi_all * L_phi_all
        )

        return {
            "total": total,
            "phys": L_phys,
            "phys_fg": L_phys_fg,
            "a": L_a,
            "C": L_C,
            "phi": L_phi,
            "phi_all": L_phi_all,
            "phi_pred": phi_pred,
        }
