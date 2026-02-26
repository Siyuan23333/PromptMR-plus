"""
Standalone evaluation script for computing SSIM, PSNR, and NMSE metrics
by comparing model inference reconstructions against ground truth.

Ground truth is computed from fully-sampled k-space data via Root Sum of
Squares (RSS) combination across coils, or loaded directly from the
dataset HDF5 files (e.g., FastMRI provides pre-computed RSS targets under
the key "reconstruction_rss").

Usage examples:

  # Compare HDF5 reconstructions against HDF5 ground truth volumes
  python evaluate.py \\
      --predictions-dir _predict/fastmri-knee/test-plus/reconstructions \\
      --targets-dir /path/to/fastMRI/knee_multicoil/multicoil_val \\
      --target-key reconstruction_rss

  # Compare HDF5 reconstructions against ground truth, auto-center-crop
  python evaluate.py \\
      --predictions-dir _predict/fastmri-knee/test-plus/reconstructions \\
      --targets-dir /path/to/fastMRI/knee_multicoil/multicoil_val \\
      --target-key reconstruction_rss \\
      --center-crop

  # Compute ground truth on-the-fly from fully-sampled k-space
  python evaluate.py \\
      --predictions-dir _predict/fastmri-knee/test-plus/reconstructions \\
      --targets-dir /path/to/fastMRI/knee_multicoil/multicoil_val \\
      --target-key kspace \\
      --compute-gt-from-kspace

  # Compare .npy reconstructions against .npy ground truth
  python evaluate.py \\
      --predictions-dir _predict/cine/reconstructions \\
      --targets-dir /path/to/cine/ground_truth \\
      --file-format npy
"""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# Metric functions (same as mri_utils/utils.py for self-contained usage)
# ---------------------------------------------------------------------------

def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((gt - pred) ** 2))


def compute_nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE).

    NMSE = ||gt - pred||^2 / ||gt||^2
    """
    return float(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def compute_psnr(gt: np.ndarray, pred: np.ndarray, maxval: float = None) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        gt: Ground truth image (2-D or 3-D array).
        pred: Predicted / reconstructed image (same shape as gt).
        maxval: Dynamic range of the images. If None, uses gt.max().
    """
    if maxval is None:
        maxval = gt.max()
    return float(peak_signal_noise_ratio(gt, pred, data_range=maxval))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, maxval: float = None) -> float:
    """Compute Structural Similarity Index (SSIM), averaged over slices.

    Expects 3-D inputs of shape [num_slices, H, W]. For a single 2-D image
    pass ``gt[np.newaxis, ...]``.

    Args:
        gt: Ground truth volume [S, H, W].
        pred: Predicted volume [S, H, W].
        maxval: Dynamic range of the images. If None, uses gt.max().
    """
    if gt.ndim == 2:
        gt = gt[np.newaxis, ...]
        pred = pred[np.newaxis, ...]
    if gt.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D arrays, got ndim={gt.ndim}")

    if maxval is None:
        maxval = gt.max()

    ssim_sum = 0.0
    for s in range(gt.shape[0]):
        ssim_sum += structural_similarity(gt[s], pred[s], data_range=maxval)
    return ssim_sum / gt.shape[0]


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def rss_from_kspace(kspace: np.ndarray) -> np.ndarray:
    """Compute RSS ground truth from fully-sampled multi-coil k-space.

    Args:
        kspace: Complex k-space array. Accepted shapes:
            - [num_slices, num_coils, H, W]  (complex64/128)
            - [num_coils, H, W]              (single slice, complex)
            - [num_coils, H, W, 2]           (real/imag in last dim)

    Returns:
        RSS magnitude image(s) of shape [num_slices, H, W] or [H, W].
    """
    # Handle real/imag last-dim representation
    if np.isrealobj(kspace) and kspace.shape[-1] == 2:
        kspace = kspace[..., 0] + 1j * kspace[..., 1]

    if kspace.ndim == 3:
        # Single slice: [coils, H, W]
        images = np.fft.ifftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
            axes=(-2, -1),
        )
        return np.sqrt(np.sum(np.abs(images) ** 2, axis=0))

    if kspace.ndim == 4:
        # Volume: [slices, coils, H, W]
        images = np.fft.ifftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
            axes=(-2, -1),
        )
        return np.sqrt(np.sum(np.abs(images) ** 2, axis=1))

    raise ValueError(f"Unexpected kspace shape: {kspace.shape}")


def center_crop(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop an image (or volume) to the target spatial dimensions.

    Args:
        image: Array of shape [..., H, W].
        target_shape: (crop_H, crop_W).
    """
    h, w = image.shape[-2], image.shape[-1]
    th, tw = target_shape
    if th > h or tw > w:
        raise ValueError(
            f"Target shape {target_shape} is larger than image shape ({h}, {w})"
        )
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return image[..., start_h : start_h + th, start_w : start_w + tw]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_h5_reconstruction(filepath: Path, key: str = "reconstruction") -> np.ndarray:
    """Load a reconstruction or target from an HDF5 file."""
    with h5py.File(filepath, "r") as hf:
        return np.array(hf[key])


def load_h5_target(filepath: Path, key: str, compute_from_kspace: bool = False) -> np.ndarray:
    """Load ground truth from an HDF5 dataset file.

    Args:
        filepath: Path to the HDF5 file.
        key: Dataset key inside the HDF5 file (e.g. "reconstruction_rss",
             "reconstruction_esc", or "kspace").
        compute_from_kspace: If True, interpret *key* as k-space data and
            compute RSS ground truth on-the-fly.
    """
    with h5py.File(filepath, "r") as hf:
        data = np.array(hf[key])
        if compute_from_kspace:
            data = rss_from_kspace(data)
        return data


def load_npy(filepath: Path) -> np.ndarray:
    """Load a .npy file."""
    return np.load(filepath)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_volume(
    pred: np.ndarray,
    target: np.ndarray,
    do_center_crop: bool = False,
) -> dict:
    """Evaluate a single volume (or 2-D image).

    Returns a dict with per-volume NMSE, PSNR, SSIM, and the number of slices.
    """
    # Make both at least 3-D: [S, H, W]
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
    if target.ndim == 2:
        target = target[np.newaxis, ...]

    # Optional center-crop to the smaller spatial extent
    if do_center_crop:
        min_h = min(pred.shape[-2], target.shape[-2])
        min_w = min(pred.shape[-1], target.shape[-1])
        pred = center_crop(pred, (min_h, min_w))
        target = center_crop(target, (min_h, min_w))

    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch after cropping: pred {pred.shape} vs target {target.shape}"
        )

    maxval = target.max()
    return {
        "nmse": compute_nmse(target, pred),
        "psnr": compute_psnr(target, pred, maxval=maxval),
        "ssim": compute_ssim(target, pred, maxval=maxval),
        "num_slices": pred.shape[0],
    }


def run_evaluation(args):
    predictions_dir = Path(args.predictions_dir)
    targets_dir = Path(args.targets_dir)

    if args.file_format == "h5":
        pred_files = sorted(predictions_dir.glob("**/*.h5"))
    else:
        pred_files = sorted(predictions_dir.glob("**/*.npy"))

    if len(pred_files) == 0:
        raise FileNotFoundError(
            f"No {args.file_format} files found in {predictions_dir}"
        )

    metrics_per_volume = {}
    agg = defaultdict(float)
    num_volumes = 0

    for pred_path in pred_files:
        # Derive the corresponding target file name
        rel = pred_path.relative_to(predictions_dir)
        if args.file_format == "h5":
            target_path = targets_dir / rel
        else:
            target_path = targets_dir / rel

        if not target_path.exists():
            print(f"WARNING: target not found for {rel}, skipping.")
            continue

        # Load prediction
        if args.file_format == "h5":
            pred = load_h5_reconstruction(pred_path, key=args.pred_key)
        else:
            pred = load_npy(pred_path)

        # Load target
        if args.file_format == "h5":
            target = load_h5_target(
                target_path,
                key=args.target_key,
                compute_from_kspace=args.compute_gt_from_kspace,
            )
        else:
            target = load_npy(target_path)

        # Evaluate
        vol_metrics = evaluate_volume(pred, target, do_center_crop=args.center_crop)
        metrics_per_volume[str(rel)] = vol_metrics

        for k in ("nmse", "psnr", "ssim"):
            agg[k] += vol_metrics[k]
        num_volumes += 1

    if num_volumes == 0:
        print("No volumes evaluated. Check your paths and file names.")
        return

    # Print per-volume results
    print(f"\n{'='*70}")
    print(f"{'Volume':<40} {'NMSE':>10} {'PSNR':>10} {'SSIM':>10}")
    print(f"{'-'*70}")
    for name, m in sorted(metrics_per_volume.items()):
        print(f"{name:<40} {m['nmse']:>10.6f} {m['psnr']:>10.4f} {m['ssim']:>10.6f}")

    # Print aggregate results
    print(f"{'='*70}")
    print(f"{'AVERAGE (' + str(num_volumes) + ' volumes)':<40} "
          f"{agg['nmse']/num_volumes:>10.6f} "
          f"{agg['psnr']/num_volumes:>10.4f} "
          f"{agg['ssim']/num_volumes:>10.6f}")
    print(f"{'='*70}\n")

    return {k: v / num_volumes for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MRI reconstructions against ground truth using SSIM, PSNR, and NMSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing model reconstruction outputs.",
    )
    parser.add_argument(
        "--targets-dir",
        type=str,
        required=True,
        help="Directory containing ground truth data (HDF5 dataset files or .npy files).",
    )
    parser.add_argument(
        "--pred-key",
        type=str,
        default="reconstruction",
        help="HDF5 key for the reconstruction in prediction files (default: 'reconstruction').",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="reconstruction_rss",
        help="HDF5 key for the ground truth in target files "
             "(default: 'reconstruction_rss'). Use 'kspace' with --compute-gt-from-kspace "
             "to derive ground truth on-the-fly.",
    )
    parser.add_argument(
        "--compute-gt-from-kspace",
        action="store_true",
        default=False,
        help="Compute RSS ground truth on-the-fly from fully-sampled k-space "
             "(requires --target-key pointing to k-space data).",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        default=False,
        help="Center-crop prediction and target to the same spatial size before computing metrics.",
    )
    parser.add_argument(
        "--file-format",
        type=str,
        choices=["h5", "npy"],
        default="h5",
        help="File format for predictions and targets (default: 'h5').",
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
