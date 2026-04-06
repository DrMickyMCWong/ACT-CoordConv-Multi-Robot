#!/usr/bin/env python3
"""Utility to overlay two images and visualize their differences.

python image_overlap.py \
  "/home/hk/Documents/ACT_Shaka/checkpoints/task1/Episode_0_real.png" \
  "/home/hk/Documents/ACT_Shaka/checkpoints/task1/Episode_0_sim.png" \
  --alpha 0.5 \
  --output-dir /home/hk/Documents/ACT_Shaka/checkpoints/task1
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


def compute_outputs(img_a: Image.Image, img_b: Image.Image, alpha: float):
    """Return blended image and heatmap overlay highlighting differences."""
    blended = Image.blend(img_a, img_b, alpha)

    diff = ImageChops.difference(img_a, img_b).convert("L")
    diff_arr = np.asarray(diff, dtype=np.float32)
    max_val = diff_arr.max()
    if max_val == 0:
        normalized = np.zeros_like(diff_arr)
    else:
        normalized = diff_arr / max_val

    heat = np.zeros((*normalized.shape, 4), dtype=np.uint8)
    heat[..., 0] = (normalized * 255).astype(np.uint8)
    heat[..., 3] = (normalized * 200).astype(np.uint8)
    heat_img = Image.fromarray(heat, mode="RGBA")

    heatmap = Image.alpha_composite(img_a, heat_img)
    return blended, heatmap


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay two screenshots and highlight differences.")
    parser.add_argument("image_a", type=Path, help="Path to the first image (base layer)")
    parser.add_argument("image_b", type=Path, help="Path to the second image")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend factor (0.0 keeps first image, 1.0 keeps second), default 0.5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store outputs (defaults to the directory of image_a)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (defaults to derived timestamp pair)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image_a.exists() or not args.image_b.exists():
        raise FileNotFoundError("Both input images must exist.")

    out_dir = args.output_dir or args.image_a.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.prefix
    if label is None:
        label = f"{args.image_a.stem}_vs_{args.image_b.stem}"

    img_a = Image.open(args.image_a).convert("RGBA")
    img_b = Image.open(args.image_b).convert("RGBA")

    if img_a.size != img_b.size:
        raise ValueError(f"Image sizes differ: {img_a.size} vs {img_b.size}.")

    blended, heatmap = compute_outputs(img_a, img_b, args.alpha)

    blend_path = out_dir / f"overlap_blend_{label}.png"
    heat_path = out_dir / f"difference_heatmap_{label}.png"

    blended.save(blend_path)
    heatmap.save(heat_path)

    print(f"Saved blend to: {blend_path}")
    print(f"Saved heatmap to: {heat_path}")


if __name__ == "__main__":
    main()
