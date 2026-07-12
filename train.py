import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .dataset import ConjFPPDataset
    from .loss import ConjLoss
    from .model import ConjUNet
    from .utils import load_checkpoint, phase_mae, save_checkpoint
except ImportError:
    from dataset import ConjFPPDataset
    from loss import ConjLoss
    from model import ConjUNet
    from utils import load_checkpoint, phase_mae, save_checkpoint


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_loaders(args, device):
    root = Path(args.data_root)
    common = dict(synthetic_root=args.synthetic_root, cache_dir=args.cache_dir)
    pin = device == "cuda"

    train_ds = ConjFPPDataset(
        fringe_dir=str(root / "train" / "fringe"),
        crop_size=args.crop_size,
        is_train=True,
        samples_per_image=args.samples_per_image,
        augment=args.augment,
        use_all_phase_steps=args.all_phase_steps,
        **common,
    )
    val_ds = ConjFPPDataset(
        fringe_dir=str(root / "val" / "fringe"),
        is_train=False,
        **common,
    )
    if args.fast:
        val_ds = Subset(val_ds, list(range(min(6, len(val_ds)))))

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        **loader_kwargs,
    )
    return train_loader, val_loader


def to_device(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]


def run_epoch(model, criterion, loader, optimizer, device, train, desc):
    model.train() if train else model.eval()
    sums = {}
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        fringe, phi_gt, mask, a_gt, C_gt_real, C_gt_imag, theta = to_device(batch, device)
        with torch.set_grad_enabled(train):
            a, Cr, Ci = model(fringe)
            out = criterion(a, Cr, Ci, fringe, theta, phi_gt, mask, a_gt, C_gt_real, C_gt_imag)
        if train:
            optimizer.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            mae_fg = phase_mae(out["phi_pred"], phi_gt, mask)
            mae_all = phase_mae(out["phi_pred"], phi_gt, None)
        scalars = {k: v for k, v in out.items() if torch.is_tensor(v) and v.ndim == 0}
        scalars["mae_fg"] = mae_fg
        scalars["mae_all"] = mae_all
        for k, v in scalars.items():
            sums[k] = sums.get(k, 0.0) + v.item()
        n += 1
        pbar.set_postfix(loss=f"{out['total'].item():.3f}", mae=f"{mae_fg.item():.4f}")

    return {k: v / max(n, 1) for k, v in sums.items()}


def score(metrics, target_fg, target_all):
    return max(metrics["mae_fg"] / target_fg, metrics["mae_all"] / target_all)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print("=" * 60)
    print("Conjugate FPP model")
    print(f"device={device}  data={args.data_root}")
    print(f"crop={args.crop_size}  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}")
    print(f"all_phase_steps={args.all_phase_steps}  lambda_C={args.lambda_C}")
    print(f"score = max(FG_MAE/{args.target_fg}, ALL_MAE/{args.target_all})")
    print("=" * 60)

    train_loader, val_loader = build_loaders(args, device)
    print(f"train patches: {len(train_loader.dataset)} | val images: {len(val_loader.dataset)}")

    model = ConjUNet(in_channels=1).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = ConjLoss(
        lambda_phys=args.lambda_phys,
        lambda_phys_fg=args.lambda_phys_fg,
        lambda_a=args.lambda_a * args.gt_scale,
        lambda_C=args.lambda_C * args.gt_scale,
        lambda_phi=args.lambda_phi,
        lambda_phi_all=args.lambda_phi_all,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        scheduler = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    latest_path = output_dir / "latest_model.pth"

    start_epoch, best_score = 0, float("inf")
    if args.resume and Path(args.resume).exists():
        start_epoch, resume_metrics = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        if resume_metrics:
            best_score = score(resume_metrics, args.target_fg, args.target_all)
        print(f"resumed at epoch {start_epoch} with score {best_score:.4f}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_m = run_epoch(model, criterion, train_loader, optimizer, device, True, f"Train {epoch + 1}")
        val_m = run_epoch(model, criterion, val_loader, optimizer, device, False, f"Val {epoch + 1}")
        val_score = score(val_m, args.target_fg, args.target_all)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_score)
        elif scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_score < best_score:
            best_score = val_score
            save_checkpoint(model, optimizer, scheduler, epoch, {**val_m, "score": val_score}, str(best_path))
            marker = "  BEST"
        save_checkpoint(model, optimizer, scheduler, epoch, {**val_m, "score": val_score}, str(latest_path))

        print(f"[{epoch + 1:3d}/{args.epochs}] ({time.time() - t0:5.1f}s) lr={lr:.1e} | "
              f"train MAE {train_m['mae_fg']:.4f} | val score {val_score:.4f} | "
              f"FG {val_m['mae_fg']:.4f} ALL {val_m['mae_all']:.4f} | "
              f"L_C {val_m['C']:.4f} L_phys {val_m['phys']:.4f}{marker}")

        if lr <= 1e-7:
            print("LR reached minimum, stopping.")
            break

    print(f"\nDone. Best score {best_score:.4f} -> {best_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Train the conjugate FPP model")
    p.add_argument("--data_root", default="../fpp-ml-bench/training_datasets/training_data_depth_individual_normalized")
    p.add_argument("--synthetic_root", default="../fpp-ml-bench/fpp_synthetic_dataset")
    p.add_argument("--cache_dir", default=".cache/conjugate_fpp_gt")
    p.add_argument("--crop_size", type=int, default=384)
    p.add_argument("--samples_per_image", type=int, default=2)
    p.add_argument("--all_phase_steps", action="store_true", default=True)
    p.add_argument("--single_phase_step", dest="all_phase_steps", action="store_false")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--lambda_phys", type=float, default=0.25)
    p.add_argument("--lambda_phys_fg", type=float, default=1.0)
    p.add_argument("--lambda_a", type=float, default=0.2)
    p.add_argument("--lambda_C", type=float, default=0.5)
    p.add_argument("--gt_scale", type=float, default=1.0)
    p.add_argument("--lambda_phi", type=float, default=4.0)
    p.add_argument("--lambda_phi_all", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine", "none"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--target_fg", type=float, default=0.08)
    p.add_argument("--target_all", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    p.add_argument("--output_dir", default="outputs/conjugate_fpp")
    p.add_argument("--fast", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
