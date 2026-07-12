import os
import uuid
import zipfile

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset


F0 = 21.0 / 960.0
N_PHASE_STEPS = 18


def wrap_to_pi(phase):
    return np.arctan2(np.sin(phase), np.cos(phase))


def _y_grid(size):
    y = np.arange(size, dtype=np.float32).reshape(size, 1)
    return np.broadcast_to(y, (size, size))


class ConjFPPDataset(Dataset):
    def __init__(self,
                 fringe_dir,
                 synthetic_root="../fpp-ml-bench/fpp_synthetic_dataset",
                 img_size=960,
                 crop_size=None,
                 is_train=False,
                 samples_per_image=1,
                 min_mask_ratio=0.015,
                 fg_center_crop_prob=0.3,
                 fg_center_jitter=160,
                 augment=False,
                 use_all_phase_steps=False,
                 cache_dir=".cache/conjugate_fpp_gt"):
        self.fringe_dir = fringe_dir
        self.synthetic_root = synthetic_root
        self.img_size = img_size
        self.f0 = F0
        self.crop_size = int(crop_size) if crop_size else None
        if self.crop_size and self.crop_size > img_size:
            raise ValueError(f"crop_size must be <= {img_size}")
        self.is_train = is_train
        self.samples_per_image = max(1, int(samples_per_image))
        self.min_mask_ratio = min_mask_ratio
        self.fg_center_crop_prob = float(fg_center_crop_prob)
        self.fg_center_jitter = max(0, int(fg_center_jitter))
        self.augment = augment
        self.phase_steps = list(range(N_PHASE_STEPS)) if use_all_phase_steps else [0]
        self.cache_dir = cache_dir
        self.y_grid = _y_grid(img_size)

        self.fringe_files = sorted(
            f for f in os.listdir(fringe_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        if not self.fringe_files:
            raise RuntimeError(f"No fringe images found in {fringe_dir}")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        if self.is_train:
            return len(self.fringe_files) * self.samples_per_image * len(self.phase_steps)
        return len(self.fringe_files)

    def _load_image(self, path):
        return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    def _load_mask(self, path):
        mask = np.loadtxt(path, delimiter=",", dtype=np.float32)
        return (mask > 0).astype(np.float32)

    def _compute_gt(self, angle_dir, obj_name, angle):
        cache = os.path.join(self.cache_dir, f"{obj_name}_{angle}.npz") if self.cache_dir else None
        if cache and os.path.exists(cache):
            try:
                with np.load(cache) as d:
                    return d["a_gt"], d["C_gt_real"], d["C_gt_imag"]
            except (OSError, EOFError, ValueError, zipfile.BadZipFile):
                try:
                    os.remove(cache)
                except OSError:
                    pass

        stack = np.stack([
            self._load_image(os.path.join(angle_dir, f"A_{k}.png"))
            for k in range(N_PHASE_STEPS)
        ], axis=0)
        a_gt = stack.mean(axis=0).astype(np.float32)

        kvec = np.arange(N_PHASE_STEPS, dtype=np.float32).reshape(N_PHASE_STEPS, 1, 1)
        C_dft = (2.0 / N_PHASE_STEPS) * (
            stack * np.exp(-2j * np.pi * kvec / N_PHASE_STEPS)
        ).sum(axis=0)
        theta = 2.0 * np.pi * self.f0 * self.y_grid
        C_gt = C_dft * np.exp(-1j * theta) / 2.0
        C_gt_real = C_gt.real.astype(np.float32)
        C_gt_imag = C_gt.imag.astype(np.float32)

        if cache:
            tmp = f"{cache}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
            np.savez_compressed(tmp, a_gt=a_gt, C_gt_real=C_gt_real, C_gt_imag=C_gt_imag)
            os.replace(tmp, cache)
        return a_gt, C_gt_real, C_gt_imag

    def _align_C_to_phi(self, C_gt_real, C_gt_imag, phi_obj, mask):
        valid = mask > 0.5
        if valid.sum() < 100:
            valid = np.ones_like(mask, dtype=bool)
        delta = wrap_to_pi(np.arctan2(C_gt_imag, C_gt_real) - phi_obj)
        offset = np.arctan2(np.sin(delta[valid]).mean(), np.cos(delta[valid]).mean()).astype(np.float32)
        cos_o, sin_o = np.cos(offset), np.sin(offset)
        return (
            (C_gt_real * cos_o + C_gt_imag * sin_o).astype(np.float32),
            (C_gt_imag * cos_o - C_gt_real * sin_o).astype(np.float32),
            offset,
        )

    def _sample_crop_origin(self, mask, c):
        limit = self.img_size - c
        if self.fg_center_crop_prob > 0 and np.random.rand() < self.fg_center_crop_prob:
            ys, xs = np.nonzero(mask > 0.5)
            if len(ys) > 0:
                i = np.random.randint(len(ys))
                cy, cx = int(ys[i]), int(xs[i])
                if self.fg_center_jitter > 0:
                    cy += np.random.randint(-self.fg_center_jitter, self.fg_center_jitter + 1)
                    cx += np.random.randint(-self.fg_center_jitter, self.fg_center_jitter + 1)
                return (
                    int(np.clip(cy - c // 2, 0, limit)),
                    int(np.clip(cx - c // 2, 0, limit)),
                )
        return int(np.random.randint(limit + 1)), int(np.random.randint(limit + 1))

    def __getitem__(self, idx):
        n = len(self.fringe_files)
        fringe_file = self.fringe_files[idx % n]
        phase_step = self.phase_steps[(idx // n) % len(self.phase_steps)] if self.is_train else 0

        obj_name, angle_png = fringe_file.rsplit("_", 1)
        angle = os.path.splitext(angle_png)[0]
        synth_dir = os.path.join(self.synthetic_root, obj_name, angle)

        if self.is_train and phase_step != 0:
            fringe = self._load_image(os.path.join(synth_dir, f"A_{phase_step}.png"))
        else:
            fringe = self._load_image(os.path.join(self.fringe_dir, fringe_file))

        wp = sio.loadmat(os.path.join(synth_dir, "wrapped_phase.mat"))["wph"].astype(np.float32)
        mask = self._load_mask(os.path.join(synth_dir, "mask.csv"))
        if mask.sum() < 100:
            mask = np.ones_like(mask)

        carrier = 2.0 * np.pi * self.f0 * self.y_grid
        phi_obj = wrap_to_pi(wp - carrier)

        a_gt, C_gt_real, C_gt_imag = self._compute_gt(synth_dir, obj_name, angle)
        C_gt_real, C_gt_imag, offset = self._align_C_to_phi(C_gt_real, C_gt_imag, phi_obj, mask)
        carrier = carrier + offset
        if self.is_train and phase_step != 0:
            carrier = carrier + 2.0 * np.pi * phase_step / N_PHASE_STEPS

        if self.crop_size and self.is_train:
            c = self.crop_size
            top, left = self._sample_crop_origin(mask, c)
            if self.min_mask_ratio > 0:
                min_pixels = max(1, int(c * c * self.min_mask_ratio))
                for _ in range(30):
                    if mask[top:top + c, left:left + c].sum() >= min_pixels:
                        break
                    top, left = self._sample_crop_origin(mask, c)
            sl = (slice(top, top + c), slice(left, left + c))
            fringe, phi_obj, mask = fringe[sl], phi_obj[sl], mask[sl]
            a_gt = a_gt[sl]
            C_gt_real, C_gt_imag, carrier = C_gt_real[sl], C_gt_imag[sl], carrier[sl]

        tensors = [fringe, phi_obj, mask, a_gt, C_gt_real, C_gt_imag, carrier]
        tensors = [torch.from_numpy(np.ascontiguousarray(t)).unsqueeze(0) for t in tensors]

        if self.augment and self.is_train:
            for dim in (-1, -2):
                if np.random.rand() > 0.5:
                    tensors = [torch.flip(t, dims=[dim]) for t in tensors]

        return tuple(tensors)
