import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0) + 1, ckpt.get("metrics", {})


def masked_mean(value, mask=None, eps=1e-6):
    if mask is None:
        return value.mean()
    return (value * mask).sum() / mask.sum().clamp(min=eps)


def phase_mae(phi_pred, phi_gt, mask=None, eps=1e-6):
    delta = phi_pred - phi_gt
    angular = torch.acos(torch.clamp(torch.cos(delta), -1.0 + eps, 1.0 - eps))
    return masked_mean(angular, mask, eps)
