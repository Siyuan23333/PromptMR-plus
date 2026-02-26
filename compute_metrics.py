"""
Compute average SSIM, PSNR, and NMSE between ground-truth and predicted
complex images stored as .npy files in shape (T, 2, H, W).

Metrics are computed on magnitude images: mag = sqrt(real^2 + imag^2).

Usage:
    python compute_metrics.py --gt_dir /path/to/gt --pred_dir /path/to/pred

Output:
    Per-subject metrics and dataset-wide averages printed to stdout.
    Optionally saved to a CSV file with --csv_out.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def to_magnitude(x):
    """Convert (T, 2, H, W) real/imag to (T, H, W) magnitude."""
    return np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)


def compute_nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def compute_psnr(gt, pred, data_range):
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)


def compute_ssim(gt, pred, data_range):
    """Compute mean SSIM across temporal frames. gt/pred shape: (T, H, W)."""
    vals = []
    for t in range(gt.shape[0]):
        vals.append(structural_similarity(gt[t], pred[t], data_range=data_range))
    return np.mean(vals)


def main():
    parser = argparse.ArgumentParser(
        description="Compute SSIM, PSNR, NMSE on (T,2,H,W) complex .npy files."
    )
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory of ground-truth .npy files (T,2,H,W)")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory of prediction .npy files (T,2,H,W)")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="Optional path to save per-subject CSV results")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    gt_files = sorted(gt_dir.glob("*.npy"))
    if not gt_files:
        print(f"No .npy files found in {gt_dir}")
        sys.exit(1)

    rows = []
    all_ssim, all_psnr, all_nmse = [], [], []

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            print(f"  SKIP {gt_path.name}: prediction not found")
            continue

        gt = np.load(str(gt_path)).astype(np.float64)      # (T, 2, H, W)
        pred = np.load(str(pred_path)).astype(np.float64)   # (T, 2, H, W)

        gt_mag = to_magnitude(gt)       # (T, H, W)
        pred_mag = to_magnitude(pred)   # (T, H, W)

        data_range = gt_mag.max()

        nmse_val = compute_nmse(gt_mag, pred_mag)
        psnr_val = compute_psnr(gt_mag, pred_mag, data_range)
        ssim_val = compute_ssim(gt_mag, pred_mag, data_range)

        all_nmse.append(nmse_val)
        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)

        rows.append({
            "subject": gt_path.stem,
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "NMSE": nmse_val,
        })
        print(f"  {gt_path.name:30s}  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}  NMSE={nmse_val:.6f}")

    if not all_ssim:
        print("No matching subject pairs found.")
        sys.exit(1)

    print("-" * 70)
    print(f"  {'AVERAGE':30s}  SSIM={np.mean(all_ssim):.4f}  PSNR={np.mean(all_psnr):.2f}  NMSE={np.mean(all_nmse):.6f}")
    print(f"  {'STD':30s}  SSIM={np.std(all_ssim):.4f}  PSNR={np.std(all_psnr):.2f}  NMSE={np.std(all_nmse):.6f}")
    print(f"  Evaluated on {len(all_ssim)} subjects.")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "SSIM", "PSNR", "NMSE"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({
                "subject": "AVERAGE",
                "SSIM": np.mean(all_ssim),
                "PSNR": np.mean(all_psnr),
                "NMSE": np.mean(all_nmse),
            })
        print(f"  Results saved to {csv_path}")


if __name__ == "__main__":
    main()
