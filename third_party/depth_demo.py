# Copyright (c) Kangan Qian. All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: 3D location estimation from 2D images using depth estimation

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from third_party.DAM.depth_anything_v2.dpt import DepthAnythingV2
from third_party.yoloworld_demo import get_2dloc_open_vocabulary_detector


def calculate_average_depth(depth_map: np.ndarray, center_pixel: tuple, neighborhood_size: int = 5) -> float:
    """
    Calculate the average depth value around a specified pixel
    
    Args:
        depth_map (np.ndarray): HxW depth map
        center_pixel (tuple): (x, y) coordinates of the center pixel
        neighborhood_size (int): Size of the neighborhood area (default: 5x5)
        
    Returns:
        float: Average depth value in the neighborhood
    """
    half_size = neighborhood_size // 2
    x, y = center_pixel
    
    # Ensure boundaries are within image dimensions
    x_start = max(0, x - half_size)
    x_end = min(depth_map.shape[1], x + half_size + 1)
    y_start = max(0, y - half_size)
    y_end = min(depth_map.shape[0], y + half_size + 1)
    
    # Extract neighborhood depths
    neighborhood_depths = depth_map[y_start:y_end, x_start:x_end]
    
    # Calculate average depth ignoring zero values
    valid_depths = neighborhood_depths[neighborhood_depths != 0]
    if valid_depths.size == 0:
        return 0.0
    
    return np.mean(valid_depths)


def pixel_to_3d_coordinates(
        depth_map: np.ndarray, 
        camera_intrinsic: np.ndarray, 
        pixel: tuple
    ) -> tuple:
    """
    Convert 2D pixel coordinates to 3D world coordinates using depth map and camera intrinsics
    
    Args:
        depth_map (np.ndarray): Depth map in meters
        camera_intrinsic (np.ndarray): 4x4 camera intrinsic matrix
        pixel (tuple): Pixel coordinates (x, y)
        
    Returns:
        tuple: 3D coordinates (X, Y, Z) in camera coordinate system
    """
    x, y = pixel
    
    # Calculate average depth in neighborhood for robustness
    avg_depth = calculate_average_depth(depth_map, pixel)
    if avg_depth != 0:
        D = avg_depth
    else:
        D = depth_map[y, x]
    
    # Extract camera intrinsic parameters
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    u0 = camera_intrinsic[0, 2]
    v0 = camera_intrinsic[1, 2]
    
    # Calculate 3D coordinates in camera frame
    X_cam = (x - u0) * D / fx
    Y_cam = (y - v0) * D / fy
    Z_cam = D
    
    return X_cam, Y_cam, Z_cam


def get_3d_location(
        text: list = ['car'], 
        image_path: str = None, 
        debug: bool = False,
        model_path: str = "./pretrained_model/depth_anything_v2_vitb.pth",
        encoder_type: str = "vitb",
        camera_intrinsic: np.ndarray = None
    ) -> tuple:
    """
    Estimate 3D locations of objects in an image
    
    Args:
        text (list): List of object names to locate
        image_path (str): Path to input image
        debug (bool): Whether to save debug visualization
        model_path (str): Path to depth estimation model
        encoder_type (str): Encoder type for depth model
        camera_intrinsic (np.ndarray): Camera intrinsic matrix
        
    Returns:
        tuple: 
            - prompt (str): Description of 3D locations
            - spatial_location (list): List of 3D coordinates for each object
    """
    # Default camera intrinsic matrix if not provided
    if camera_intrinsic is None:
        camera_intrinsic = np.array([
            [1.25281310e+03, 0.00000000e+00, 8.26588115e+02, 0.00000000e+00],
            [0.00000000e+00, 1.25281310e+03, 4.69984663e+02, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
    
    # Configure depth model based on encoder type
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize and load depth estimation model
    model = DepthAnythingV2(**model_configs[encoder_type]).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and process image
    raw_img = cv2.imread(image_path)
    depth_map = model.infer_image(raw_img)  # HxW depth map
    
    # Invert depth map if needed (depends on model output)
    max_depth = np.max(depth_map)
    depth_map = max_depth - depth_map
    
    # Save debug visualization if requested
    if debug:
        plt.imshow(depth_map, cmap='jet')
        plt.colorbar()
        plt.title("Predicted Depth Map")
        plt.savefig("./debug/debug_depth.png")
    
    # Process each object
    prompt = ""
    spatial_location = []
    for obj in text:
        # Get 2D location using open-vocabulary detector
        loc2d_prompt, location_2d = get_2dloc_open_vocabulary_detector(
            text=[obj], 
            image_path=image_path
        )
        
        prompt += loc2d_prompt
        
        # Handle case where 2D location not found
        if location_2d is None:
            prompt += f"\nFailed to estimate 3D location for {obj}. You must infer or identify it yourself."
            continue
        
        # Convert to integer pixel coordinates
        pixel = [int(round(coord)) for coord in location_2d]
        
        # Calculate 3D coordinates
        X, Y, Z = pixel_to_3d_coordinates(depth_map, camera_intrinsic, pixel)
        spatial_location.append([X, Y, Z])
        
        prompt += f"\nEstimated 3D location(x,y,z) for {obj} in camera coordinates: [{X:.2f}, {Y:.2f}, {Z:.2f}], z={Z:.2f}"
    
    prompt += f"""(Note: The Z coordinate represents depth, with smaller Z-values indicating closer proximity to the camera/front.)"""
    return prompt, spatial_location


if __name__ == '__main__':
    # Example usage
    image_path = "./third_party/nuscenes_CAM_FRONT_5978.webp"
    model_path = "./pretrained_model/depth_anything_v2_vitb.pth"
    
    # Camera intrinsic matrix
    camera_intrinsic = np.array([
        [1.25281310e+03, 0.00000000e+00, 8.26588115e+02, 0.00000000e+00],
        [0.00000000e+00, 1.25281310e+03, 4.69984663e+02, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    
    # Initialize depth model
    model = DepthAnythingV2(
        encoder='vitb',
        features=128,
        out_channels=[96, 192, 384, 768]
    ).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load image and estimate depth
    raw_img = cv2.imread(image_path)
    depth_map = model.infer_image(raw_img)
    
    # Test conversion for a specific pixel
    pixel = (1164, 627)
    X, Y, Z = pixel_to_3d_coordinates(depth_map, camera_intrinsic, pixel)
    
    print(f"3D coordinates in camera frame: X = {X:.3f} m, Y = {Y:.3f} m, Z = {Z:.3f} m")