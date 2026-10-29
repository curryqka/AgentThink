# Copyright (c) 2025 The AgentThink Authors. All rights reserved.
"""Lightweight visual tool library.

Contains only the single-view utilities (open-vocabulary detection, monocular
depth, resize, crop).  It is used by the DriveMLLM demo, where no nuScenes
perception pickle is available.  The driving-centric tools live in
:mod:`scripts.tools.tool_libraries`.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance

from third_party.depth_demo import get_3d_location
from third_party.yoloworld_demo import get_2dbox_open_vocabulary_detector


class FuncAgent:
    """Bind the visual tools to one image / camera list."""

    def __init__(self, data_dict=None, json_data_dict=None) -> None:
        self.data = data_dict
        self.json_data_dict = json_data_dict
        self.short_trajectory_description = False

        self.visual_func_infos = [
            get_open_world_vocabulary_detection_info,
            get_3d_loc_in_cam_info,
            resize_image_info,
            crop_image_info,
        ]

    # -- tool implementations ------------------------------------------------
    def get_open_world_vocabulary_detection(self, object_names, image_path):
        """Tool name used by the catalog (paper Appendix A)."""
        return self.get_open_world_vocabulary_detection_info(object_names, image_path)

    def get_open_world_vocabulary_detection_info(self, object_names, image_path):
        return get_2dbox_open_vocabulary_detector(
            text=object_names, image_path=image_path
        )

    def get_3d_loc_in_cam(self, object_names, image_path):
        """Tool name used by the catalog (paper Appendix A)."""
        return self.get_3d_loc_in_cam_info(object_names, image_path)

    def get_3d_loc_in_cam_info(self, object_names, image_path):
        return get_3d_location(text=object_names, image_path=image_path)

    def resize_image(self, input_path, output_path, target_size, interpolation=Image.BILINEAR):
        return resize_image(input_path, output_path, target_size, interpolation)

    def crop_image(self, input_path, output_path, box):
        return crop_image(input_path, output_path, box)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

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
                "description": "Target dimensions [width, height]",
            },
            "interpolation": {
                "type": "integer",
                "description": "Interpolation method (e.g., Image.BILINEAR)",
            },
        },
        "required": ["input_path", "output_path", "target_size"],
    },
}


def resize_image(input_path, output_path, target_size, interpolation=Image.BILINEAR):
    with Image.open(input_path) as img:
        img.resize(target_size, interpolation).save(output_path)


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
                "description": "Crop region [left, upper, right, lower]",
            },
        },
        "required": ["input_path", "output_path", "box"],
    },
}


def crop_image(input_path, output_path, box):
    with Image.open(input_path) as img:
        img.crop(box).save(output_path)


rotate_image_info = {
    "name": "rotate_image",
    "description": "Rotates an image by specified degrees with canvas expansion support",
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {"type": "string", "description": "Input image file path"},
            "output_path": {"type": "string", "description": "Output path for rotated image"},
            "degrees": {"type": "number", "description": "Rotation angle in degrees"},
            "expand": {"type": "boolean", "description": "Whether to expand the canvas"},
            "fill_color": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "description": "RGB fill color for expanded areas",
            },
        },
        "required": ["input_path", "output_path", "degrees"],
    },
}


def rotate_image(input_path, output_path, degrees, expand=False, fill_color=(255, 255, 255)):
    with Image.open(input_path) as img:
        img.rotate(degrees, expand=expand, fillcolor=fill_color).save(output_path)


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
                "description": "Brightness multiplier (1.0=original, >1.0=brighter, <1.0=darker)",
            },
        },
        "required": ["input_path", "output_path", "factor"],
    },
}


def adjust_brightness(input_path, output_path, factor):
    with Image.open(input_path) as img:
        ImageEnhance.Brightness(img).enhance(factor).save(output_path)


# ---------------------------------------------------------------------------
# Vision-language tools
# ---------------------------------------------------------------------------

get_open_world_vocabulary_detection_info = {
    "name": "get_open_world_vocabulary_detection",
    "description": (
        "Detects objects in an image using open-vocabulary detection and returns "
        "their 2D bounding boxes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "object_names": {"type": "list", "description": "List of objects to detect"},
            "image_path": {"type": "str", "description": "Path to the image file"},
        },
        "required": ["object_names", "image_path"],
    },
}


get_3d_loc_in_cam_info = {
    "name": "get_3d_loc_in_cam",
    "description": "Calculates 3D locations of objects in camera coordinates",
    "parameters": {
        "type": "object",
        "properties": {
            "object_names": {"type": "list", "description": "List of objects to locate"},
            "image_path": {"type": "str", "description": "Path to the image file"},
        },
        "required": ["object_names", "image_path"],
    },
}
