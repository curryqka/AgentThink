# Copyright (c) Kangan Qian. All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: Function agent for autonomous driving visual processing

from pathlib import Path
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon
from third_party.yoloworld_demo import get_2dbox_open_vocabulary_detector
from third_party.depth_demo import get_3d_location


class FuncAgent:
    def __init__(self, data_dict=None, json_data_dict=None) -> None:
        """
        Initialize function agent for visual processing tasks
        
        Args:
            data_dict: Dictionary containing scene data
            json_data_dict: Dictionary containing JSON metadata
        """
        self.data = data_dict
        self.json_data_dict = json_data_dict
        self.short_trajectory_description = False

        # Define available visual functions
        self.visual_func_infos = [
            get_open_world_vocabulary_detection_info,
            get_3d_loc_in_cam_info,
            resize_image_info,
            crop_image_info,
        ]

    def get_open_world_vocabulary_detection(self, object_names: list, cam_type: str):
        """
        Detect objects in an image using open vocabulary detection
        
        Args:
            object_names: List of objects to detect
            cam_type: Camera type to process
            
        Returns:
            Tuple of prompts and detected bounding boxes
        """
        cam_path_info_list = self.json_data_dict['image']
        for cam_path_info in cam_path_info_list:
            if cam_type == cam_path_info.split('/')[1]:
                cur_cam_type_index = cam_path_info_list.index(cam_path_info)
        
        choosed_image_path = cam_path_info_list[cur_cam_type_index]
        prompts, detected_2d_boxs = get_2dbox_open_vocabulary_detector(
            text=object_names, 
            image_path=choosed_image_path
        )

        return prompts, detected_2d_boxs 

    def get_open_world_vocabulary_detection_info(self, object_names: list, image_path: str):
        """
        Detect objects in an image using open vocabulary detection
        
        Args:
            object_names: List of objects to detect
            image_path: Path to the image file
            
        Returns:
            Tuple of prompts and detected bounding boxes
        """
        prompts, detected_2d_boxs = get_2dbox_open_vocabulary_detector(
            text=object_names, 
            image_path=image_path
        )
        return prompts, detected_2d_boxs 

    def get_3d_loc_in_cam_info(self, object_names: list, image_path: str):
        """
        Get 3D locations of objects in camera coordinates
        
        Args:
            object_names: List of objects to locate
            image_path: Path to the image file
            
        Returns:
            Tuple of prompts and 3D locations
        """
        prompts, detected_loc_3d = get_3d_location(
            text=object_names, 
            image_path=image_path
        )
        return prompts, detected_loc_3d

    def get_ego_states(self):
        """Get ego vehicle state information"""
        return get_ego_prompts(self.data)


# Image processing functions and their metadata
resize_image_info = {
    "name": "resize_image",
    "description": "Resizes an image to specified dimensions with interpolation support",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {"type": "string", "description": "Input image file path"},
            "output_path": {"type": "string", "description": "Output path for resized image"},
            "target_size": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Target dimensions [width, height]"
            },
            "interpolation": {
                "type": "integer",
                "description": "Interpolation method (e.g., Image.BILINEAR for bilinear interpolation)"
            }
        },
        "required": ["input_path", "output_path", "target_size"]
    }
}


def resize_image(input_path, output_path, target_size, interpolation=Image.BILINEAR):
    """
    Resize an image to specified dimensions
    
    Args:
        input_path: Path to input image file
        output_path: Path to save resized image
        target_size: Target dimensions (width, height)
        interpolation: Interpolation method (default: bilinear)
    """
    with Image.open(input_path) as img:
        resized_img = img.resize(target_size, interpolation)
        resized_img.save(output_path)


crop_image_info = {
    "name": "crop_image",
    "description": "Crops a rectangular region from an image",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {"type": "string", "description": "Input image file path"},
            "output_path": {"type": "string", "description": "Output path for cropped image"},
            "box": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Crop region coordinates [left, upper, right, lower]"
            }
        },
        "required": ["input_path", "output_path", "box"]
    }
}


def crop_image(input_path, output_path, box):
    """
    Crop a region from an image
    
    Args:
        input_path: Path to input image file
        output_path: Path to save cropped image
        box: Crop region coordinates (left, upper, right, lower)
    """
    with Image.open(input_path) as img:
        cropped_img = img.crop(box)
        cropped_img.save(output_path)


rotate_image_info = {
    "name": "rotate_image",
    "description": "Rotates an image by specified degrees with canvas expansion support",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {"type": "string", "description": "Input image file path"},
            "output_path": {"type": "string", "description": "Output path for rotated image"},
            "degrees": {"type": "number", "description": "Rotation angle in degrees (clockwise)"},
            "expand": {
                "type": "boolean",
                "description": "Whether to expand canvas to fit rotation (default: False)"
            },
            "fill_color": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": "RGB fill color for expanded areas (default: [255,255,255])"
            }
        },
        "required": ["input_path", "output_path", "degrees"]
    }
}


def rotate_image(input_path, output_path, degrees, expand=False, fill_color=(255, 255, 255)):
    """
    Rotate an image by specified degrees
    
    Args:
        input_path: Path to input image file
        output_path: Path to save rotated image
        degrees: Rotation angle in degrees
        expand: Whether to expand canvas to fit rotation
        fill_color: Fill color for expanded areas
    """
    with Image.open(input_path) as img:
        rotated_img = img.rotate(degrees, expand=expand, fillcolor=fill_color)
        rotated_img.save(output_path)


adjust_brightness_info = {
    "name": "adjust_brightness",
    "description": "Adjusts image brightness using enhancement factor",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {"type": "string", "description": "Input image file path"},
            "output_path": {"type": "string", "description": "Output path for adjusted image"},
            "factor": {
                "type": "number",
                "description": "Brightness multiplier (1.0=original, >1.0=brighter, <1.0=darker)"
            }
        },
        "required": ["input_path", "output_path", "factor"]
    }
}


def adjust_brightness(input_path, output_path, factor):
    """
    Adjust image brightness
    
    Args:
        input_path: Path to input image file
        output_path: Path to save adjusted image
        factor: Brightness multiplier (1.0=original, >1.0=brighter, <1.0=darker)
    """
    with Image.open(input_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        bright_img.save(output_path)


get_open_world_vocabulary_detection_info = {
    "name": "get_open_world_vocabulary_detection",
    "description": "Detects objects in an image using open vocabulary detection",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "list",
                "description": "List of objects to detect",
            },
            "image_path": {
                "type": "str",
                "description": "Path to the image file"
            }
        },
        "required": ["text", "image_path"],
    },
}


get_3d_loc_in_cam_info = {
    "name": "get_3d_loc_in_cam",
    "description": "Calculates 3D locations of objects in camera coordinates",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "list",
                "description": "List of objects to locate",
            },
            "image_path": {
                "type": "str",
                "description": "Path to the image file"
            }
        },
        "required": ["text", "image_path"],
    },
}