"""
Convert multi-coil fully-sampled k-space data into 2-channel complex images
using precomputed sensitivity maps for coil combination.

Input:
    ksp_dir/  contains  {subject}.npy   shape (T, C, H, W) complex64
    sens_dir/ contains  {subject}.npy   shape (T, C, H, W) complex64
                                     or shape (C, H, W)    complex64  (time-invariant)

Output:
    out_dir/  contains  {subject}.npy   shape (T, 2, H, W) float32
              channel 0 = real, channel 1 = imaginary

Coil combination per frame:
    img = IFFT2c(kspace)                       # (C, H, W) complex
    combined = sum_c( img_c * conj(sens_c) )   # (H, W)   complex
    output = stack(real, imag)                  # (2, H, W) float32
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mri_utils import ifft2c, complex_mul, complex_conj


def coil_combine_frame(kspace_frame, sens_frame):
    """
    Combine one temporal frame using sensitivity maps.

    Args:
        kspace_frame: (C, H, W) complex64 numpy array
        sens_frame:   (C, H, W) complex64 numpy array

    Returns:
        (2, H, W) float32 numpy array  [real, imag]
    """
    # to torch with real/imag split: (C, H, W, 2)
    ksp = torch.from_numpy(
        np.stack([kspace_frame.real, kspace_frame.imag], axis=-1)
    ).float()
    sens = torch.from_numpy(
        np.stack([sens_frame.real, sens_frame.imag], axis=-1)
    ).float()

    # IFFT to image domain: (C, H, W, 2)
    img = ifft2c(ksp)

    # coil combine: sum over C of img * conj(sens)
    combined = complex_mul(img, complex_conj(sens)).sum(dim=0)  # (H, W, 2)

    # (H, W, 2) -> (2, H, W)
    return combined.permute(2, 0, 1).numpy()


def convert_subject(ksp_path, sens_path, out_path):
    kspace = np.load(str(ksp_path))          # (T, C, H, W) complex
    sens_maps = np.load(str(sens_path))      # (T, C, H, W) or (C, H, W) complex

    time_invariant = (sens_maps.ndim == 3)
    T = kspace.shape[0]
    frames = []

    for t in range(T):
        sens_t = sens_maps if time_invariant else sens_maps[t]
        frame = coil_combine_frame(kspace[t], sens_t)   # (2, H, W)
        frames.append(frame)

    result = np.stack(frames, axis=0).astype(np.float32)  # (T, 2, H, W)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), result)
    print(f"  {out_path.name}  shape={result.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert multi-coil k-space to 2-channel complex images "
                    "using precomputed sensitivity maps."
    )
    parser.add_argument("--ksp_dir", type=str, required=True,
                        help="Directory of fully-sampled k-space .npy files (T,C,H,W) complex")
    parser.add_argument("--sens_dir", type=str, required=True,
                        help="Directory of sensitivity map .npy files (T,C,H,W) or (C,H,W) complex")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for (T,2,H,W) .npy files")
    parser.add_argument("--split_json", type=str, default=None,
                        help="Optional split JSON; if given, only process subjects in --partition")
    parser.add_argument("--partition", type=str, default="val",
                        help="Which partition to process (default: val)")
    args = parser.parse_args()

    ksp_dir = Path(args.ksp_dir)
    sens_dir = Path(args.sens_dir)
    out_dir = Path(args.out_dir)

    # determine subject list
    if args.split_json:
        with open(args.split_json) as f:
            subjects = json.load(f)[args.partition]
    else:
        subjects = sorted([p.stem for p in ksp_dir.glob("*.npy")])

    print(f"Processing {len(subjects)} subjects -> {out_dir}")
    for subj in subjects:
        ksp_path = ksp_dir / f"{subj}.npy"
        sens_path = sens_dir / f"{subj}.npy"
        if not ksp_path.exists():
            print(f"  SKIP {subj}: k-space not found at {ksp_path}")
            continue
        if not sens_path.exists():
            print(f"  SKIP {subj}: sens map not found at {sens_path}")
            continue
        convert_subject(ksp_path, sens_path, out_dir / f"{subj}.npy")

    print("Done.")


if __name__ == "__main__":
    main()
