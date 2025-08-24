# Copyright (c). All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: Script for generating image collages from JSON data

import os
import pickle
import json
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def check_folder(dir_path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        dir_path (str): Path to the directory to check/create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")


def remove_last_part(s: str) -> str:
    """Remove the last part of a string after the last underscore.
    
    Args:
        s (str): Input string to process.
        
    Returns:
        str: Processed string with last part removed.
    """
    last_underscore_index = s.rfind('_')
    return s[:last_underscore_index] if last_underscore_index != -1 else s


def check_saved_path(save_path: str, gt_path: str) -> bool:
    """Check if save path matches ground truth path.
    
    Args:
        save_path (str): Generated save path.
        gt_path (str): Ground truth path.
        
    Returns:
        bool: True if paths match, False otherwise.
    """
    return save_path == gt_path


def main(args: argparse.Namespace) -> None:
    """Main function to process JSON data and create image collages.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Load JSON file
    with open(args.json_file_path, 'r') as f:
        data_list = json.load(f)
    
    # Process each entry in JSON data
    for data in tqdm(data_list, desc="Processing JSON entries"):
        # Get list of image paths
        image_paths = data['image']
        
        # Load images
        images = []
        for img_path in image_paths:
            # Construct full image path
            full_img_path = os.path.join(args.image_root_dir, img_path)
            # Load image
            image = cv2.imread(full_img_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Warning: Could not load image {full_img_path}")
        
        # Skip if no valid images found
        if not images:
            print(f"No valid images found for entry with idx {data['idx']}. Skipping.")
            continue
        
        # Get image dimensions (assuming all images have same size)
        height, width, _ = images[0].shape
        
        # Create blank canvas for collage (2 rows, 3 columns layout)
        collage = np.zeros((2 * height, 3 * width, 3), dtype=np.uint8)
        
        # Arrange images on the canvas
        for i, img in enumerate(images):
            row = i // 3
            col = i % 3
            collage[row * height:(row + 1) * height, col * width:(col + 1) * width] = img
        
        # Create output directory if it doesn't exist
        if not os.path.exists(args.savepath_img):
            os.makedirs(args.savepath_img)
        
        # Generate output filename
        image_name = remove_last_part(data['idx'])
        save_path = os.path.join(args.savepath_img, f"{image_name}.png")
        
        # Save collage image
        cv2.imwrite(save_path, collage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save a collage of six images based on a JSON file.')
    parser.add_argument('--json_file_path', type=str, 
                        default="/path/to/DriveLMMo1_TRAIN.json", 
                        help='Path to the JSON file')
    parser.add_argument('--image_root_dir', type=str, 
                        default="/path/to/nuscenes", 
                        help='Root directory where the images are stored')
    parser.add_argument('--savepath_img', type=str, 
                        default="/path/to/output/directory", 
                        help='Directory where the collage image will be saved')
    args = parser.parse_args()
    main(args)