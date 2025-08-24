# Copyright (c) Kangan Qian. All rights reserved.
# Authors: Kangan Qian (Tsinghua University, Xiaomi Corporation)
# Description: Open-vocabulary object detection using YOLO-World

from ultralytics import YOLO
from typing import List, Tuple, Union
import os


def get_2dbox_open_vocabulary_detector(
        text: Union[str, List[str]] = ['car'], 
        image_path: str = None,
        model_path: str = "./pretrained_models/yolov8x-worldv2.pt"
    ) -> Tuple[str, Union[None, List]]:
    """
    Detect objects in an image using open-vocabulary detection and return 2D bounding boxes
    
    Args:
        text (str|list): Object names to detect (single string or list of strings)
        image_path (str): Path to input image file
        model_path (str): Path to YOLO-World model checkpoint
        
    Returns:
        Tuple containing:
        - prompt (str): Description of detection results
        - box_2d (list|None): List of detected bounding boxes or None if none found
    """
    # Ensure text is a list
    if not isinstance(text, list):
        text = [text]
    
    # Initialize YOLO-World model
    model = YOLO(model_path)
    
    # Set classes to detect
    model.set_classes(text)
    
    # Execute prediction
    results = model.predict(image_path)
    
    # Extract detection results
    box_cls = results[0].boxes.cls
    xyxy = results[0].boxes.xyxy
    box_2d = xyxy.cpu().tolist()
    
    # Handle case where no objects are detected
    if not box_2d:
        prompt = ""
        for obj_name in text:
            prompt += f"\nFailed to detect 2D bounding box for {obj_name}. You must infer or identify it yourself."
        return prompt, None
    
    # Build prompt with detection results
    prompt = ""
    for obj_name in text:
        if obj_name not in model.names.values():
            prompt += f"\nFailed to detect 2D bounding box for {obj_name}. You must infer or identify it yourself."
            continue
        
        # Find the class ID for this object name
        class_id = [k for k, v in model.names.items() if v == obj_name][0]
        
        if class_id not in box_cls.cpu().tolist():
            prompt += f"\nFailed to detect 2D bounding box for {obj_name}. You must infer or identify it yourself."
            continue
        
        # Get the bounding box for this object
        box_index = box_cls.cpu().tolist().index(class_id)
        prompt += f"\nDetected 2D bounding box for {obj_name}: {box_2d[box_index]}."
    
    return prompt, box_2d[0] if box_2d else None


def get_2dloc_open_vocabulary_detector(
        text: Union[str, List[str]] = ['car'], 
        image_path: str = None,
        model_path: str = "./pretrained_models/yolov8x-worldv2.pt"
    ) -> Tuple[str, Union[None, List]]:
    """
    Detect objects in an image using open-vocabulary detection and return 2D locations
    
    Args:
        text (str|list): Object names to detect (single string or list of strings)
        image_path (str): Path to input image file
        model_path (str): Path to YOLO-World model checkpoint
        
    Returns:
        Tuple containing:
        - prompt (str): Description of detection results
        - pixel_location (list|None): List of detected pixel locations or None if none found
    """
    # Ensure text is a list
    if not isinstance(text, list):
        text = [text]
    
    # Initialize YOLO-World model
    model = YOLO(model_path)
    
    # Set classes to detect
    model.set_classes(text)
    
    # Execute prediction
    results = model.predict(image_path)
    
    # Extract detection results
    box_cls = results[0].boxes.cls
    xywh = results[0].boxes.xywh
    pixel_location = xywh.cpu().numpy()[:, 0:2].tolist()
    
    # Handle case where no objects are detected
    if not pixel_location:
        prompt = ""
        for obj_name in text:
            prompt += f"\nFailed to detect 2D location for {obj_name}. You must infer or identify it yourself."
        return prompt, None
    
    # Build prompt with detection results
    prompt = ""
    for obj_name in text:
        if obj_name not in model.names.values():
            prompt += f"\nFailed to detect 2D location for {obj_name}. You must infer or identify it yourself."
            continue
        
        # Find the class ID for this object name
        class_id = [k for k, v in model.names.items() if v == obj_name][0]
        
        if class_id not in box_cls.cpu().tolist():
            prompt += f"\nFailed to detect 2D location for {obj_name}. You must infer or identify it yourself."
            continue
        
        # Get the location for this object
        loc_index = box_cls.cpu().tolist().index(class_id)
        prompt += f"\nDetected 2D location for {obj_name}: {pixel_location[loc_index]}."
    
    return prompt, pixel_location[0] if pixel_location else None


if __name__ == '__main__':
    # Example usage
    image_path = "./third_party/nuscenes_CAM_FRONT_5976.webp"
    model_path = ".//yolov8x-worldv2.pt"
    objects_to_detect = ["black motorcycle", 'silver car']
    
    # Detect bounding boxes
    box_prompt, detected_box = get_2dbox_open_vocabulary_detector(
        text=objects_to_detect,
        image_path=image_path,
        model_path=model_path
    )
    print("Bounding Box Detection Results:")
    print(box_prompt)
    
    # Detect locations
    loc_prompt, detected_location = get_2dloc_open_vocabulary_detector(
        text=objects_to_detect,
        image_path=image_path,
        model_path=model_path
    )
    print("\nLocation Detection Results:")
    print(loc_prompt)