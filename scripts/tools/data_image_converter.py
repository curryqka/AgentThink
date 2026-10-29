#!/usr/bin/env python
# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Stitch the six nuScenes cameras into one image per sample.

The DriveLMM-o1 JSON stores six camera paths per sample.  AgentThink consumes a
single 3 x 2 collage whose filename is ``<idx without question suffix>.png``.

Example:
    python scripts/tools/data_image_converter.py \
        --json-file-path data/DriveLMMo1/DriveLMMo1_TEST.json \
        --image-root-dir /path/to/nuscenes \
        --savepath-img data/image2concat
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def image_name(idx: str) -> str:
    """Map ``scene_frame_question`` to the collage filename (no extension)."""
    return idx.rsplit("_", 1)[0]


def make_collage(images, rows: int = 2, cols: int = 3) -> np.ndarray:
    """Arrange camera images row-major on a blank canvas."""
    height, width = images[0].shape[:2]
    collage = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for index, image in enumerate(images[: rows * cols]):
        row, col = divmod(index, cols)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        collage[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return collage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build six-view collages")
    parser.add_argument("--json-file-path", required=True, help="DriveLMMo1_TRAIN/TEST.json")
    parser.add_argument("--image-root-dir", required=True, help="nuScenes root directory")
    parser.add_argument("--savepath-img", required=True, help="Output directory")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip samples with unreadable images instead of failing.")
    args = parser.parse_args()

    with open(args.json_file_path, "r", encoding="utf-8") as handle:
        data_list = json.load(handle)
    os.makedirs(args.savepath_img, exist_ok=True)

    written = skipped = 0
    for data in tqdm(data_list, desc="Building collages"):
        images = []
        for relative_path in data["image"]:
            image = cv2.imread(os.path.join(args.image_root_dir, relative_path))
            if image is None:
                print(f"Warning: could not read {relative_path}")
                continue
            images.append(image)

        if len(images) < 6:
            skipped += 1
            if not args.skip_missing:
                raise FileNotFoundError(
                    f"Sample {data['idx']} has only {len(images)}/6 readable images. "
                    "Use --skip-missing to ignore such samples."
                )
            continue

        save_path = os.path.join(args.savepath_img, f"{image_name(data['idx'])}.png")
        cv2.imwrite(save_path, make_collage(images))
        written += 1

    print(f"Wrote {written} collages to {args.savepath_img} (skipped {skipped}).")


if __name__ == "__main__":
    main()
