# Agent for functional calls
# Written by Jiageng Mao, Modified by Kangan Qian

from pathlib import Path
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from scripts.tools.utils.geometry import CAR_LENGTH, CAR_WIDTH, GRID_SIZE, rotate_bbox
from skimage.draw import polygon
from scripts.tools.utils.box_distance import polygons_overlap, polygon_distance
from scripts.tools.utils.geometry import location_to_pixel_coordinate, pixel_coordinate_to_location, GRID_SIZE
from third_party.yoloworld_demo import get_2dbox_open_vocabulary_detector
from third_party.depth_demo import get_3d_location

class FuncAgent:
    def __init__(self, data_dict=None, json_data_dict=None) -> None:
        self.data = data_dict

        # add json data dict to load image meta data
        self.json_data_dict = json_data_dict
        self.short_trajectory_description = False

        self.visual_func_infos = [
            get_open_world_vocabulary_detection_info,
            get_3d_loc_in_cam_info,
            resize_image_info,
            crop_image_info,

        ]
        self.detection_func_infos = [
            get_leading_object_detection_info,
            # get_object_detections_in_range_info,
            get_surrounding_object_detections_info,
            # get_front_object_detections_info,
            get_all_object_detections_info,
        ]
        self.prediction_func_infos = [
            get_leading_object_future_trajectory_info,
            get_future_trajectories_for_specific_objects_info,
            # get_future_trajectories_in_range_info,
            # get_future_waypoint_of_specific_objects_at_timestep_info,
            get_all_future_trajectories_info,
        ]
        self.occupancy_func_infos = [
            get_occupancy_at_locations_for_timestep_info,
            # check_occupancy_for_planned_trajectory_info,
        ]
        self.map_func_infos = [
            get_drivable_at_locations_info,
            # check_drivable_of_planned_trajectory_info,
            get_lane_category_at_locations_info,
            get_distance_to_shoulder_at_locations_info,
            get_current_shoulder_info,
            get_distance_to_lane_divider_at_locations_info,
            get_current_lane_divider_info,
            get_nearest_pedestrian_crossing_info,
        ]
    """Basic visual information"""
    def get_open_world_vocabulary_detection(self, object_names:list, cam_type:str):
        """
            "idx": "e7ef871f77f44331aefdebc24ec034b7_b10f0cd792b64d16a1a5e8349b20504c_3",
        "image": [
            "samples/CAM_FRONT_LEFT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_LEFT__1533201471404844.jpg",
            "samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201471412477.jpg",
            "samples/CAM_FRONT_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_RIGHT__1533201471420339.jpg",
            "samples/CAM_BACK_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_BACK_RIGHT__1533201471427893.jpg",
            "samples/CAM_BACK/n015-2018-08-02-17-16-37+0800__CAM_BACK__1533201471437636.jpg",
            "samples/CAM_BACK_LEFT/n015-2018-08-02-17-16-37+0800__CAM_BACK_LEFT__1533201471447423.jpg"
        """
        cam_path_info_list = self.json_data_dict['image']
        for cam_path_info in cam_path_info_list:
            if cam_type == cam_path_info.split('/')[1]:
                cur_cam_type_index = cam_path_info_list.index(cam_path_info)
        
        choosed_image_path = cam_path_info_list[cur_cam_type_index]

        prompts, detected_2d_boxs = get_2dbox_open_vocabulary_detector(text=object_names, image_path=choosed_image_path)

        return prompts, detected_2d_boxs 

    

    """Detection functions""" 
    def get_leading_object_detection(self):
        return get_leading_object_detection(self.data)
    
    def get_surrounding_object_detections(self):
        return get_surrounding_object_detections(self.data)
    
    def get_front_object_detections(self):
        return get_front_object_detections(self.data)
    
    def get_object_detections_in_range(self, x_start, x_end, y_start, y_end):
        return get_object_detections_in_range(x_start, x_end, y_start, y_end, self.data)
    
    def get_all_object_detections(self):
        return get_all_object_detections(self.data)
    
    """Prediction functions"""
    def get_leading_object_future_trajectory(self):
        return get_leading_object_future_trajectory(self.data, short=self.short_trajectory_description)
    
    def get_future_trajectories_for_specific_objects(self, object_ids):
        return get_future_trajectories_for_specific_objects(object_ids, self.data, short=self.short_trajectory_description)
    
    def get_future_trajectories_in_range(self, x_start, x_end, y_start, y_end):
        return get_future_trajectories_in_range(x_start, x_end, y_start, y_end, self.data, short=self.short_trajectory_description)
        
    def get_future_waypoint_of_specific_objects_at_timestep(self, object_ids, timestep):
        return get_future_waypoint_of_specific_objects_at_timestep(object_ids, timestep, self.data)
    
    def get_all_future_trajectories(self):
        return get_all_future_trajectories(self.data, short=self.short_trajectory_description)
    
    """Occupancy functions"""
    def get_occupancy_at_locations_for_timestep(self, locations, timestep):
        return get_occupancy_at_locations_for_timestep(locations, timestep, self.data)
    
    def check_occupancy_for_planned_trajectory(self, trajectory):
        return check_occupancy_for_planned_trajectory(trajectory, self.data)

    """Map functions"""
    def get_drivable_at_locations(self, locations):
        return get_drivable_at_locations(locations, self.data)
    
    def check_drivable_of_planned_trajectory(self, trajectory):
        return check_drivable_of_planned_trajectory(trajectory, self.data)
    
    def get_lane_category_at_locations(self, locations, return_score=True):
        return get_lane_category_at_locations(locations, self.data, return_score=return_score)
    
    def get_distance_to_shoulder_at_locations(self, locations):
        return get_distance_to_shoulder_at_locations(locations, self.data)
    
    def get_current_shoulder(self):
        return get_current_shoulder(self.data)
    
    def get_distance_to_lane_divider_at_locations(self, locations):
        return get_distance_to_lane_divider_at_locations(locations, self.data)
    
    def get_current_lane_divider(self):
        return get_current_lane_divider(self.data)
    
    def get_nearest_pedestrian_crossing(self): 
        return get_nearest_pedestrian_crossing(self.data)  
    
    """Ego-state functions"""
    def get_ego_states(self):
        return get_ego_prompts(self.data)

# visual functions
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
    调整图像尺寸
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param target_size: 目标尺寸 (width, height)
    :param interpolation: 插值方法（默认双线性插值）
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
    裁剪图像区域
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param box: 裁剪区域 (left, upper, right, lower)
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

def rotate_image(input_path, output_path, degrees, expand=False, fill_color=(255,255,255)):
    """
    旋转图像
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param degrees: 旋转角度
    :param expand: 是否扩展画布
    :param fill_color: 扩展区域的填充颜色
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
    调整图像亮度
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param factor: 亮度系数1.0为原始，>1.0更亮，<1.0更暗
    """
    with Image.open(input_path) as img:
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        bright_img.save(output_path)

get_open_world_vocabulary_detection_info ={
    "name": "get_open_world_vocabulary_detection",
    "description": "Given a list of words of object(e.g.['car', 'bike', 'traffic light']), get the detection of the object, the function will return its 2d position and size within the camera coordinate system. If there is no text-related object, return None",
    "parameters": {
      "type": "object",
      "properties": {
            "text": {
                "type": "list",
                "description": "a list contains the words to detect",
            },
            "image_path": {
                "type": "str",
                "description": "the image path related to the image."
            }
        },
        "required": ["text", "image_path"],
    },
}


get_3d_loc_in_cam_info ={
    "name": "get_3d_loc_in_cam",
    "description": "Given an input image and a set of object-related keywords(specified by a List of object words, e.g.['pedestrian in red T-shirt', 'black SUV']), this function calculates the depth value for each pixel in the image. It then determines and returns the 3D location of the specified objects within the camera coordinate system.",
    "parameters": {
      "type": "object",
      "properties": {
            "text": {
                "type": "list",
                "description": "a list contains the words to detect",
            },
            "image_path": {
                "type": "str",
                "description": "the image path related to the image."
            }
        },
        "required": ["text", "image_path"],
    },
}


# Detection functions

get_leading_object_detection_info ={
    "name": "get_leading_object_detection",
    "description": "Get the detection of the leading object, the function will return the leading object id and its position and size. If there is no leading object, return None",
    "parameters": {
      "type": "object",
      "properties": {

        },
        "required": [],
    },
}

def get_leading_object_detection(data_dict):
    objects = data_dict["objects"]
    prompts = "Leading object detections:\n"
    detected_objs = []
    for obj in objects:
        # search for the leading object (at the same lane and in front of the ego vehicle in 10m)
        obj_x, obj_y = obj["bbox"][:2]
        if abs(obj_x) < 3.0 and obj_y >= 0.0 and obj_y < 10.0:
            prompts += f"Leading object detected, object type: {obj['name']}, object id: {obj['id']}, position: ({obj_x:.2f}, {obj_y:.2f}), size: ({obj['bbox'][3]:.2f}, {obj['bbox'][4]:.2f})\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

get_surrounding_object_detections_info = {
    "name": "get_surrounding_object_detections",
    "description": "Get the detections of the surrounding objects in a 20m*20m range, the function will return a list of surroundind object ids and their positions and sizes. If there is no surrounding object, return None",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_surrounding_object_detections(data_dict):
    objects = data_dict["objects"]
    prompts = "Surrounding object detections:\n"
    detected_objs = []

    for obj in objects:
        # search for the surrounding objects (20m*20m range)
        obj_x, obj_y = obj["bbox"][:2]
        if abs(obj_x) < 20.0 and abs(obj_y) < 20.0:
            prompts += f"Surrounding object detected, object type: {obj['name']}, object id: {obj['id']}, position: ({obj_x:.2f}, {obj_y:.2f}), size: ({obj['bbox'][3]:.2f}, {obj['bbox'][4]:.2f})\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None

    # breakpoint()
    return prompts, detected_objs

get_front_object_detections_info = {
    "name": "get_front_object_detections",
    "description": "Get the detections of the objects in front of you in a 10m*20m range, the function will return a list of front object ids and their positions and sizes. If there is no front object, return None",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_front_object_detections(data_dict):
    objects = data_dict["objects"]
    prompts = "Front object detections:\n"
    detected_objs = []
    for obj in objects:
        # search for the front objects (10m*20m range)
        obj_x, obj_y = obj["bbox"][:2]
        if abs(obj_x) < 5.0 and obj_y >= 0.0 and obj_y < 20.0:
            prompts += f"Front object detected, object type: {obj['name']}, object id: {obj['id']}, position: ({obj_x:.2f}, {obj_y:.2f}), size: ({obj['bbox'][3]:.2f}, {obj['bbox'][4]:.2f})\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

get_object_detections_in_range_info = {
    "name": "get_object_detections_in_range",
    "description": "Get the detections of the objects in a given range (x_start, x_end)*(y_start, y_end)m^2, the function will return a list of object ids and their positions and sizes. If there is no object, return None",
    "parameters": {
        "type": "object",
        "properties": {
            "x_start": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "start range of x axis",
            },
            "x_end": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "end range of x axis",
            },
            "y_start": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "start range of y axis",
            },
            "y_end": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "end range of y axis",
            },
        },
        "required": ["x_start", "x_end", "y_start", "y_end"],
    },
}

def get_object_detections_in_range(x_start, x_end, y_start, y_end, data_dict):
    x_start, x_end, y_start, y_end = float(x_start), float(x_end), float(y_start), float(y_end)
    objects = data_dict["objects"]
    prompts = f"Object detections in X range {x_start:.2f}-{x_end:.2f} and Y range {y_start:.2f}-{y_end:.2f}:\n"
    detected_objs = []
    for obj in objects:
        # search for the objects in range
        obj_x, obj_y = obj["bbox"][:2]
        if obj_x >= x_start and obj_x <= x_end and obj_y >= y_start and obj_y <= y_end:
            prompts += f"Object detected, object type: {obj['name']}, object id: {obj['id']}, position: ({obj_x:.2f}, {obj_y:.2f}), size: ({obj['bbox'][3]:.2f}, {obj['bbox'][4]:.2f})\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

get_all_object_detections_info ={
    "name": "get_all_object_detections",
    "description": "Get the detections of all objects in the whole scene, the function will return a list of object ids and their positions and sizes. Always avoid using this function if there are other choices.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_all_object_detections(data_dict):
    objects = data_dict["objects"]
    prompts = f"Full object detections:\n"
    detected_objs = []
    for obj in objects:
        obj_x, obj_y = obj["bbox"][:2]
        prompts += f"Object detected, object type: {obj['name']}, object id: {obj['id']}, position: ({obj_x:.2f}, {obj_y:.2f}), size: ({obj['bbox'][3]:.2f}, {obj['bbox'][4]:.2f})\n"
        detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

def check_rotate_object_collision_for_planned_trajectory(trajectory, data_dict, safe_margin=1.):
    objects = data_dict["objects"]

    if debug:
        plt.figure()

    agents_final_corners = []
    for obj in objects:
        x, y, z, dx, dy, dz, rotation_z, rotation_y, rotation_x = obj["bbox"]
        cx, cy = x, y

        rotated_corners = rotate_bbox(0, 0, dx, dy, rotation_z)

        # Get the box corners
        if debug:
            plt.scatter(x+dx/2, y+dy/2, c='b', s=50)
            plt.scatter(x+dx/2, y-dy/2, c='b', s=50)
            plt.scatter(x-dx/2, y-dy/2, c='b', s=50)
            plt.scatter(x-dx/2, y+dy/2, c='b', s=50)
            plt.show()

            final_corners_0 = [(cx + x_prime, cy + y_prime) for x_prime, y_prime in rotated_corners]

            for pt in final_corners_0:
                plt.scatter(pt[0], pt[1], c='g', s=50)

            for pt in obj["traj"]: ## NOTE: traj consists of the center of the bbox
                plt.scatter(pt[0], pt[1], c='r', s=100)
        
        agent_final_corners = []    
        for pt in obj["traj"][:6]:
            cx, cy = pt[0], pt[1]
            agent_final_corners.append([(cx + x_prime, cy + y_prime) for x_prime, y_prime in rotated_corners]) # only save future 6 timesteps

            if debug:
                for pt in agent_final_corners[-1]:
                    plt.scatter(pt[0], pt[1], c='g', s=50)
        
        agents_final_corners.append(agent_final_corners)

    ego_final_corners = []
    for pt in trajectory:
        ego_cx, ego_cy = pt[0], pt[1]
        ego_rotated_corners = rotate_bbox(ego_cx, ego_cy, CAR_WIDTH, CAR_LENGTH, 0) # NOTE ego vehicle is always facing front in evaluation, we can consider to rotate it in practice
        ego_final_corners.append(ego_rotated_corners) # only save future 6 timesteps
        
        if debug:
            for pt in ego_final_corners[-1]:
                plt.scatter(pt[0], pt[1], c='r', s=50)

    collision = np.full(len(trajectory), False)
    for ts in range(len(trajectory)):
        for obj in agents_final_corners:
            if polygons_overlap(ego_final_corners[ts], obj[ts]):
                collision[ts] = True
                if debug:
                    print("Collision detected")
                    plt.figure()
                    plt.scatter(ego_final_corners[ts][:, 0], ego_final_corners[ts][:, 1], c='b', s=50)
                    plt.scatter(np.array(obj[ts])[:,0], np.array(obj[ts])[:,1], c='y', s=50)

            elif polygon_distance(ego_final_corners[ts], obj[ts]) < safe_margin:
                collision[ts] = True
                if debug:
                    print("Collision detected")
                    plt.figure()
                    plt.scatter(ego_final_corners[ts][:, 0], ego_final_corners[ts][:, 1], c='b', s=50)
                    plt.scatter(np.array(obj[ts])[:,0], np.array(obj[ts])[:,1], c='y', s=50)

    return collision

"""LANE"""
LANE_CATEGORYS = ['divider', 'ped_crossing', 'boundary']

get_drivable_at_locations_info = {
    "name": "get_drivable_at_locations",
    "description": "Get the drivability at the locations [(x_1, y_1), ..., (x_n, y_n)]. If the location is out of the map scope, return None",
    "parameters": {
        "type": "object",
        "properties":  {
            "locations": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "the locations [(x_1, y_1), ..., (x_n, y_n)] to be queried"
            },
        },
        "required": ["locations"]
    }
}

def get_drivable_at_locations(locations, data_dict):
    drivable_map = data_dict["map"]["drivable"].T
    prompts = "Drivability of selected locations:\n"
    drivable = []
    for x, y in locations:
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid:
            drivable.append(True)
            continue
        else:
            if drivable_map[X, Y]:
                prompts += f"Location ({x:.2f}, {y:.2f}) is drivable\n"
            else:
                prompts += f"Location ({x:.2f}, {y:.2f}) is not drivable\n"
            drivable.append(drivable_map[X, Y])
    return prompts, drivable

check_drivable_of_planned_trajectory_info = {
    "name": "check_drivable_of_planned_trajectory",
    "description": "Check the drivability at the planned trajectory",
    "parameters": {
        "type": "object",
        "properties":  {
            "trajectory": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "minItems": 6,
                "maxItems": 7,
                "description": "the planned trajectory [(x_1, y_1), ..., (x_n, y_n)] to be queried",
            },
        },
        "required": ["trajectory"],
    },
}

def check_drivable_of_planned_trajectory(trajectory, data_dict):
    drivable_map = data_dict["map"]["drivable"].T
    prompts = "Drivability of the planned trajectory:\n"
    drivable = []
    all_drivable = True
    for timestep, waypoint in enumerate(trajectory):
        x, y = waypoint
        T = timestep + 1 # assume time step starting from 1
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid:
            drivable.append(True)
        else:
            if not drivable_map[X, Y]:
                prompts += f"Waypoint ({x:.2f}, {y:.2f}) is not drivable at time step {T}\n"
                all_drivable = False
            drivable.append(drivable_map[X, Y])
    if all_drivable:
        prompts += f"All waypoints of the planned trajectory are in drivable regions\n"
    return prompts, drivable

def check_drivable_of_planned_trajectory_and_surrounding(trajectory, data_dict):
    drivable_map = data_dict["map"]["drivable"].T
    prompts = "Drivability of the planned trajectory:\n"
    drivable = []
    all_drivable = True
    for timestep, waypoint in enumerate(trajectory):
        x, y = waypoint
        T = timestep + 1 # assume time step starting from 1
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid:
            drivable.append(True)
        else:
            if not drivable_map[X, Y]:
                prompts += f"Waypoint ({x:.2f}, {y:.2f}) is not drivable at time step {T}\n"

                # check surrounding
                surrounding = drivable_map[X-1:X+2, Y-1:Y+2]
                if True in surrounding:
                    index_x, index_y = np.where(surrounding)
                    index_X, index_Y = index_x + X - 1, index_y + Y - 1
                    prompts += f"- Surrounding drivable regions: {[(pixel_coordinate_to_location(x, y)[:-1]) for x, y in zip(index_X, index_Y)]}\n"
                all_drivable = False
            drivable.append(drivable_map[X, Y])
    if all_drivable:
        prompts += f"All waypoints of the planned trajectory are in drivable regions\n"
    return prompts, drivable

get_lane_category_at_locations_info = {
    "name": "get_lane_category_at_locations",
    "description": "Get the lane category at the locations [(x_1, y_1), ..., (x_n, y_n)]. If the location is out of the map scope, return None",
    "parameters": {
        "type": "object",
        "properties":  {
            "locations": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "the locations [(x_1, y_1), ..., (x_n, y_n)] to be queried",
            },
            "return_score": {
                "type": "boolean",
                "description": "whether to return the probability score of the lane category",
            },
        },
        "required": ["locations", "return_score"],
    },
}

def get_lane_category_at_locations(locations, data_dict, return_score=True):
    lane_map = data_dict["map"]["lane"].transpose(0, 2, 1)
    lane_score_map = data_dict["map"]["lane_probs"].transpose(0, 2, 1)
    prompts = "Lane category of selected locations:\n"
    lane_category = []
    for x, y in locations:
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid:
            lane_category.append(None)
            continue
        else:
            lane_category.append(lane_map[:, X, Y])
            cat_index = np.where(lane_map[:, X, Y])[0]
            if len(cat_index) == 0:
                prompts += f"Location ({x:.2f}, {y:.2f}) has no lane category\n"
            else:
                cat_prompt = ', '.join(LANE_CATEGORYS[i] for i in cat_index)
                score_prompt = ', '.join(f"{lane_score_map[i, X, Y]:.2f}" for i in cat_index)
                if return_score:
                    prompts += f"Location ({x:.2f}, {y:.2f}) has lane category {cat_prompt} with probability score {score_prompt}\n"
                else:
                    prompts += f"Location ({x:.2f}, {y:.2f}) has lane category {cat_prompt}\n"
    return prompts, lane_category

get_distance_to_shoulder_at_locations_info = {
    "name": "get_distance_to_shoulder_at_locations",
    "description": "Get the distance to both sides of road shoulders at the locations [(x_1, y_1), ..., (x_n, y_n)]. If the location is out of the map scope, return None",
    "parameters": {
        "type": "object",
        "properties":  {
            "locations": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "the locations [(x_1, y_1), ..., (x_n, y_n)] to be queried",
            },
        },
        "required": ["locations"],
    },
}

def get_distance_to_shoulder_at_locations(locations, data_dict):
    boundary_map = data_dict["map"]["lane"][2].T
    prompts = "Distance to both sides of road shoulders of selected locations:\n"
    distance_to_shoulder = []
    # breakpoint()
    for x, y in locations:
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid:
            distance_to_shoulder.append(None)
            continue
        else:
            Y_max = Y+5 if Y+5 < boundary_map.shape[1] else boundary_map.shape[1] - 1
            Y_min = Y-5 if Y-5 >= 0 else 0
            # find left nearest boundary
            ind_x = np.where(boundary_map[:X, Y_min:Y_max])[0]
            if len(ind_x) == 0:
                left_shoulder = None
            else:
                left_shoulder = (X - np.max(ind_x)) * GRID_SIZE
            # find right nearest boundary
            ind_x = np.where(boundary_map[X:, Y_min : Y_max])[0] + X
            if len(ind_x) == 0:
                right_shoulder = None
            else:
                right_shoulder = (np.min(ind_x) - X) * GRID_SIZE
            distance_to_shoulder.append((left_shoulder, right_shoulder))
            if left_shoulder is not None and right_shoulder is not None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to left shoulder is {left_shoulder}m and right shoulder is {right_shoulder}m\n"
            elif left_shoulder is None and right_shoulder is None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to shoulders are uncertain\n"
            elif left_shoulder is None and right_shoulder is not None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to left shoulder is uncertain and distance to right shoulder is {right_shoulder}m\n"
            elif left_shoulder is not None and right_shoulder is None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to left shoulder is {left_shoulder}m and distance to right shoulder is uncertain\n"
            else:
                raise Exception("Should not reach here")
    return prompts, distance_to_shoulder


get_current_shoulder_info = {
    "name": "get_current_shoulder",
    "description": "Get the distance to both sides of road shoulders for the current ego-vehicle location.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_current_shoulder(data_dict):
    boundary_map = data_dict["map"]["lane"][2].T
    prompts = "Distance to both sides of road shoulders of current ego-vehicle location:\n"
    distance_to_shoulder = []
    x, y = 0.0, 0.0
    X, Y, valid = location_to_pixel_coordinate(x, y)
    if not valid:
        distance_to_shoulder.append(None)
        prompts = None
    else:
        Y_max = Y+5 if Y+5 < boundary_map.shape[1] else boundary_map.shape[1] - 1
        Y_min = Y-5 if Y-5 >= 0 else 0
        # find left nearest boundary
        ind_x = np.where(boundary_map[:X, Y_min:Y_max])[0]
        if len(ind_x) == 0:
            left_shoulder = None
        else:
            left_shoulder = (X - np.max(ind_x)) * GRID_SIZE
        # find right nearest boundary
        ind_x = np.where(boundary_map[X:, Y_min : Y_max])[0] + X
        if len(ind_x) == 0:
            right_shoulder = None
        else:
            right_shoulder = (np.min(ind_x) - X) * GRID_SIZE
        distance_to_shoulder.append((left_shoulder, right_shoulder))
        if left_shoulder is not None and right_shoulder is not None:
            prompts += f"Current ego-vehicle's distance to left shoulder is {left_shoulder}m and right shoulder is {right_shoulder}m\n"
        elif left_shoulder is None and right_shoulder is None:
            prompts += f"Current ego-vehicle's distance to shoulders are uncertain\n"
        elif left_shoulder is None and right_shoulder is not None:
            prompts += f"Current ego-vehicle's distance to left shoulder is uncertain and distance to right shoulder is {right_shoulder}m\n"
        elif left_shoulder is not None and right_shoulder is None:
            prompts += f"Current ego-vehicle's distance to left shoulder is {left_shoulder}m and distance to right shoulder is uncertain\n"
        else:
            raise Exception("Should not reach here")
    return prompts, distance_to_shoulder

# TODO(Jiageng): add this function

# get_current_center_line_info = {
#     "name": "get_current_center_line",
#     "description": "Get the current center line that the ego-vehicle is driving on. If there is no such lane, return None",
#     "parameters": {
#     },
# }

get_distance_to_lane_divider_at_locations_info = {
    "name": "get_distance_to_lane_divider_at_locations",
    "description": "Get the distance to both sides of road lane_dividers at the locations [(x_1, y_1), ..., (x_n, y_n)]. If the location is out of the map scope, return None",
    "parameters": {
        "type": "object",
        "properties":  {
            "locations": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "the locations [(x_1, y_1), ..., (x_n, y_n)] to be queried",
            },
        },
        "required": ["locations"],
    },
}

def get_distance_to_lane_divider_at_locations(locations, data_dict):
    boundary_map = data_dict["map"]["lane"][0].T
    prompts = "Get distance to both sides of road lane_dividers of selected locations:\n"
    distance_to_lane_divider = []
    for x, y in locations:
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid:
            distance_to_lane_divider.append(None)
            continue
        else:
            Y_max = Y+5 if Y+5 < boundary_map.shape[1] else boundary_map.shape[1] - 1
            Y_min = Y-5 if Y-5 >= 0 else 0
            # find left nearest lane divider
            ind_x = np.where(boundary_map[:X, Y_min:Y_max])[0]
            if len(ind_x) == 0:
                left_lane_divider = None
            else:
                left_lane_divider = (X - np.max(ind_x)) * GRID_SIZE
            # find right nearest lane divider
            ind_x = np.where(boundary_map[X:, Y_min : Y_max])[0] + X
            if len(ind_x) == 0:
                right_lane_divider = None
            else:
                right_lane_divider = (np.min(ind_x) - X) * GRID_SIZE
            distance_to_lane_divider.append((left_lane_divider, right_lane_divider))
            if left_lane_divider is not None and right_lane_divider is not None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to left lane_divider is {left_lane_divider}m and right lane_divider is {right_lane_divider}m\n"
            elif left_lane_divider is None and right_lane_divider is None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to lane_dividers are uncertain\n"
            elif left_lane_divider is None and right_lane_divider is not None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to left lane_divider is uncertain and distance to right lane_divider is {right_lane_divider}m\n"
            elif left_lane_divider is not None and right_lane_divider is None:
                prompts += f"Location ({x:.2f}, {y:.2f}) distance to left lane_divider is {left_lane_divider}m and distance to right lane_divider is uncertain\n"
            else:
                raise Exception("Should not reach here")
    return prompts, distance_to_lane_divider

get_current_lane_divider_info = {
    "name": "get_current_lane_divider",
    "description": "Get the distance to both sides of road lane_dividers for the current ego-vehicle location",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_current_lane_divider(data_dict):
    boundary_map = data_dict["map"]["lane"][0].T
    prompts = "Get distance to both sides of road lane_dividers of current ego-vehicle location:\n"
    distance_to_lane_divider = []
    x, y = 0.0, 0.0
    X, Y, valid = location_to_pixel_coordinate(x, y)
    if not valid:
        distance_to_lane_divider.append(None)
        prompts = None
    else:
        Y_max = Y+5 if Y+5 < boundary_map.shape[1] else boundary_map.shape[1] - 1
        Y_min = Y-5 if Y-5 >= 0 else 0
        # find left nearest lane divider
        ind_x = np.where(boundary_map[:X, Y_min:Y_max])[0]
        if len(ind_x) == 0:
            left_lane_divider = None
        else:
            left_lane_divider = (X - np.max(ind_x)) * GRID_SIZE
        # find right nearest lane divider
        ind_x = np.where(boundary_map[X:, Y_min : Y_max])[0] + X
        if len(ind_x) == 0:
            right_lane_divider = None
        else:
            right_lane_divider = (np.min(ind_x) - X) * GRID_SIZE
        distance_to_lane_divider.append((left_lane_divider, right_lane_divider))
        if left_lane_divider is not None and right_lane_divider is not None:
            prompts += f"Current ego-vehicle's distance to left lane_divider is {left_lane_divider}m and distance to right lane_divider is {right_lane_divider}m\n"
        elif left_lane_divider is None and right_lane_divider is None:
            prompts += f"Current ego-vehicle's distance to both lane_dividers are uncertain\n"
        elif left_lane_divider is None and right_lane_divider is not None:
            prompts += f"Current ego-vehicle's distance to left lane_divider is uncertain and distance to right lane_divider is {right_lane_divider}m\n"
        elif left_lane_divider is not None and right_lane_divider is None:
            prompts += f"Current ego-vehicle's distance to left lane_divider is {left_lane_divider}m and distance to right lane_divider is uncertain\n"
        else:
            raise Exception("Should not reach here")
    return prompts, distance_to_lane_divider

get_nearest_pedestrian_crossing_info = {
    "name": "get_nearest_pedestrian_crossing",
    "description": "Get the location of the nearest pedestrian crossing to the ego-vehicle. If there is no such pedestrian crossing, return None",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_nearest_pedestrian_crossing(data_dict):
    boundary_map = data_dict["map"]["lane"][1].T
    prompts = "Get the nearest pedestrian crossing location:\n"
    distance_to_nearest_pedestrian_crossing = []
    X, Y, valid = location_to_pixel_coordinate(0.0, 0.0)
    if not valid:
        prompts = None
        return prompts, distance_to_nearest_pedestrian_crossing
    else:
        ind_X, ind_Y = np.where(boundary_map[:, Y:]) # Plz double check this
        ind_Y += Y
        if len(ind_X) == 0:
            prompts = None
            return prompts, distance_to_nearest_pedestrian_crossing
        else:
            dist = np.abs(ind_X - X) ** 2 + np.abs(ind_Y - Y) ** 2
            ind = np.argmin(dist)
            min_ped_crossing_X, min_ped_crossing_Y = ind_X[ind], ind_Y[ind]
            min_ped_crossing_x, min_ped_crossing_y, _ = pixel_coordinate_to_location(min_ped_crossing_X, min_ped_crossing_Y)
            prompts += f"The nearest pedestrian crossing is at ({min_ped_crossing_x:.2f}, {min_ped_crossing_y:.2f})\n"
            distance_to_nearest_pedestrian_crossing.append((min_ped_crossing_x, min_ped_crossing_y))
    return prompts, distance_to_nearest_pedestrian_crossing


get_leading_object_future_trajectory_info = {
    "name": "get_leading_object_future_trajectory",
    "description": "Get the predicted future trajectory of the leading object, the function will return a trajectory containing a series of waypoints. If there is no leading vehicle, return None",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_leading_object_future_trajectory(data_dict, short=False):
    objects = data_dict["objects"]
    prompts = "Leading object future trajectory:\n"
    detected_objs = []
    for obj in objects:
        # search for the leading object (at the same lane and in front of the ego vehicle in 10m)
        obj_x, obj_y = obj["bbox"][:2]
        if abs(obj_x) < 3.0 and obj_y >= 0.0 and obj_y < 10.0:
            if short:
                prompts += f"Leading object found, object type: {obj['name']}, object id: {obj['id']}, moving to: ({obj['traj'][5, 0]:.2f}, {obj['traj'][5, 1]:.2f})\n"
            else:
                trajectory_points = ', '.join(f"({x:.2f}, {y:.2f})" for x, y in obj['traj'][:6])
                prompts += f"Leading object found, object type: {obj['name']}, object id: {obj['id']}, future waypoint coordinates in 6s: [{trajectory_points}]\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

get_future_trajectories_for_specific_objects_info = {
    "name": "get_future_trajectories_for_specific_objects",
    "description": "Get the future trajectories of specific objects (specified by a List of object ids, e.g.['o1', 'o2', 'o3']), the function will return trajectories for each object. If there is no object, return None",
    "parameters": {
        "type": "object",
        "properties": {
            "object_ids": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 0,
                },
                "description": "a list of integer object ids",
            },
        },
        "required": ["object_ids"],
    },
}

def get_future_trajectories_for_specific_objects(object_ids, data_dict, short=False):
    objects = data_dict["objects"]
    prompts = "Future trajectories for specific objects:\n"
    detected_objs = []
    for obj in objects:
        # breakpoint()

        if 'o'+str(obj["id"]) in object_ids:
            if short:
                prompts += f"Object type: {obj['name']}, object id: {obj['id']}, moving to: ({obj['traj'][5, 0]:.2f}, {obj['traj'][5, 1]:.2f})\n"
            else:
                trajectory_points = ', '.join(f"({x:.2f}, {y:.2f})" for x, y in obj['traj'][:6])
                prompts += f"Object type: {obj['name']}, object id: {obj['id']}, future waypoint coordinates in 3s: [{trajectory_points}]\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

get_future_trajectories_in_range_info = {
    "name": "get_future_trajectories_in_range",
    "description": "Get the future trajectories where any waypoint in this trajectory falls into a given range (x_start, x_end)*(y_start, y_end)m^2, the function will return each trajectory that satisfies the condition. If there is no trajectory satisfied, return None",
    "parameters": {
        "type": "object",
        "properties": {
            "x_start": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "start range of x axis",
            },
            "x_end": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "end range of x axis",
            },
            "y_start": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "start range of y axis",
            },
            "y_end": {
                "type": "number",
                "minimum": -50,
                "maximum": 50,
                "multipleOf" : 0.01,
                "description": "end range of y axis",
            },
        },
        "required": ["x_start", "x_end", "y_start", "y_end"],
    },
}

def get_future_trajectories_in_range(x_start, x_end, y_start, y_end, data_dict, short=False):
    objects = data_dict["objects"]
    prompts = f"Future trajectories in X range {x_start:.2f}-{x_end:.2f} and Y range {y_start:.2f}-{y_end:.2f}:\n"
    detected_objs = []
    for obj in objects:
        # search for the objects in range
        obj_x, obj_y = obj["bbox"][:2]
        if obj_x >= x_start and obj_x <= x_end and obj_y >= y_start and obj_y <= y_end:
            if short:
                prompts += f"Object found, object type: {obj['name']}, object id: {obj['id']}, moving to: ({obj['traj'][5, 0]:.2f}, {obj['traj'][5, 1]:.2f})\n"
            else:
                trajectory_points = ', '.join(f"({x:.2f}, {y:.2f})" for x, y in obj['traj'][:6])
                prompts += f"Object found, object type: {obj['name']}, object id: {obj['id']}, future waypoint coordinates in 3s: [{trajectory_points}]\n"
            detected_objs.append(obj)
    if len(detected_objs) == 0:
        prompts = None
    return prompts, detected_objs

get_future_waypoint_of_specific_objects_at_timestep_info = {
    "name": "get_future_waypoint_of_specific_objects_at_timestep",
    "description": "Get the future waypoints of specific objects at a specific timestep, the function will return a list of waypoints. If there is no object or the object does not have a waypoint at the given timestep, return None",
    "parameters": {
        "type": "object",
        "properties": {
            "object_ids": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 0,
                },
                "description": "a list of object ids",
            },
            "timestep": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "multipleOf" : 1,
                "description": "the selected timestep of the future trajectory, integer value range [1-6]",
            },
        },
        "required": ["object_ids", "timestep"],
    },
}

def get_future_waypoint_of_specific_objects_at_timestep(object_ids, timestep, data_dict):
    objects = data_dict["objects"]
    prompts = f"Future waypoints of specific objects at time {timestep/2 + 0.5}s:\n"
    detected_objs = []
    for object_id in object_ids:
        obj = objects[object_id]
        if len(obj["traj"]) > timestep:
            prompts += f"object type: {obj['name']}, object id: {obj['id']}, waypoint: ({obj['traj'][timestep, 0]:.2f}, {obj['traj'][timestep, 1]:.2f}) at timestep {timestep}\n"
        else:
            prompts = None
        detected_objs.append(obj)
    if len(prompts) == 0:
        prompts = None
    return prompts, detected_objs

get_all_future_trajectories_info = {
    "name": "get_all_future_trajectories",
    "description": "Get the predicted future trajectories of all objects in the whole scene, the function will return a list of object ids and their future trajectories. Always avoid using this function if there are other choices.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
    },
}

def get_all_future_trajectories(data_dict, short=False):
    objects = data_dict["objects"]
    prompts = "All future trajectories:\n"
    for obj in objects:
        if short:
            prompts += f"Object type: {obj['name']}, object id: {obj['id']}, moving to: ({obj['traj'][5, 0]:.2f}, {obj['traj'][5, 1]:.2f})\n"
        else:
            trajectory_points = ', '.join(f"({x:.2f}, {y:.2f})" for x, y in obj['traj'][:6])
            prompts += f"Object type: {obj['name']}, object id: {obj['id']}, future waypoint coordinates in 3s: [{trajectory_points}]\n"
    if len(objects) == 0:
        prompts = None
    return prompts, objects


OCC_TH = 0.1

get_occupancy_at_locations_for_timestep_info = {
    "name": "get_occupancy_at_locations_for_timestep",
    "description": "Get the probability whether a list of locations [(x_1, y_1), ..., (x_n, y_n)] is occupied at the timestep t. If the location is out of the occupancy prediction scope, return None",
    "parameters": {
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "occupancy at the locations [(x_1, y_1), ..., (x_n, y_n)]",
            },
            "timestep": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
                "multipleOf" : 1,
                "description": "time step t in the occupancy flow, must be one of [0, 1, 2, 3, 4], which denotes the future occupancy at [0s, 0.5s, 1s, 1.5s, 2s].",
            },
        },
        "required": ["locations", "timestep"],
    },
}

def get_occupancy_at_locations_for_timestep(locations, timestep, data_dict):
    occ_map = data_dict["occupancy"].transpose(0, 2, 1)
    occ_list = []
    prompts = "Occupancy information:\n"

    for location in locations:
        x, y = location 
        X, Y, valid = location_to_pixel_coordinate(x, y)
        T = timestep
        # deal with exceptions
        if not valid or T < 0 or T >= 5:
            prompts = None
            return prompts, False

        occ = occ_map[T, X, Y]
        if occ > OCC_TH:
            prompts += f"Location ({x:.2f}, {y:.2f}) is occupied at timestep {timestep}\n"
        else:
            prompts += f"Location ({x:.2f}, {y:.2f}) is not occupied at timestep {timestep}\n"
        occ_list.append(occ)
    return prompts, occ_list


check_occupancy_for_planned_trajectory_info = {
    "name": "check_occupancy_for_planned_trajectory",
    "description": "Evaluate whether the planned trajectory [(x_1, y_1), ..., (x_n, y_n)] collides with other objects.",
    "parameters": {
        "type": "object",
        "properties": {
            "trajectory": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "minItems": 2,
                    "maxItems": 2
                },
                "minItems": 6,
                "maxItems": 7,
                "description": "the planned trajectory [(x_1, y_1), ..., (x_n, y_n)]",
            },
        },
        "required": ["trajectory"],
    },
}

def check_occupancy_for_planned_trajectory(trajectory, data_dict):
    occ_map = data_dict["occupancy"].transpose(0, 2, 1)
    prompts = "Check collision of the planned trajectory:\n"

    collision = False
    for timestep, location in enumerate(trajectory):
        x, y = location 
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid: # trajectory out of range
            continue
        T = timestep + 1 # We assume the time step starting from 1
        if T >= 5:
            continue

        occ = occ_map[T, X, Y]
        if occ > OCC_TH:
            prompts += f"Waypoint ({x:.2f}, {y:.2f}) collides at timestep {T}\n"
            collision = True
        else:
            continue
    if not collision:
        prompts += f"The planned trajectory does not collide with any other objects.\n"
    return prompts, collision

def check_occupancy_for_planned_trajectory_and_surrounding(trajectory, data_dict):
    occ_map = data_dict["occupancy"].transpose(0, 2, 1)
    prompts = "Check collision of the planned trajectory:\n"

    collision = False
    for timestep, location in enumerate(trajectory):
        x, y = location 
        X, Y, valid = location_to_pixel_coordinate(x, y)
        if not valid: # trajectory out of range
            continue
        T = timestep + 1 # We assume the time step starting from 1
        if T >= 5:
            continue

        occ = occ_map[T, X, Y]
        if occ > OCC_TH:
            prompts += f"Waypoint ({x:.2f}, {y:.2f}) collides at timestep {T}\n"

            # check surrounding
            surrounding = occ_map[T, X-1:X+2, Y-1:Y+2]
            if True in surrounding:
                index_x, index_y = np.where(surrounding)
                index_X, index_Y = index_x + X - 1, index_y + Y - 1
                prompts += f"- Surrounding not occupied region: {[(pixel_coordinate_to_location(x, y)[:-1]) for x, y in zip(index_X, index_Y)]}\n"
            collision = True
        else:
            continue
    if not collision:
        prompts += f"The planned trajectory does not collide with any other objects.\n"
    return prompts, collision

def check_collision(car_length, car_width, trajectory, occ_map):
        pts = np.array([
                [-car_length / 2. + 0.5, car_width / 2.],
                [car_length / 2. + 0.5, car_width / 2.],
                [car_length / 2. + 0.5, -car_width / 2.],
                [-car_length / 2. + 0.5, -car_width / 2.],
            ])    

        pts = (pts - (- MAP_METER)) / GRID_SIZE
        pts[:, [0, 1]] = pts[:, [1, 0]]

        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1) # all points inside the box (car)

        n_future = occ_map.shape[0] # trajectory.shape[0]   since we only have 4 future occupancy

        trajectory = trajectory * np.array([-1, 1])
        trajectory = trajectory[:, np.newaxis, :] # (n_future, 1, 2)

        trajectory[:,:,[0,1]] = trajectory[:,:,[1,0]]
        trajectory = trajectory / GRID_SIZE
        trajectory = trajectory + rc # (n_future, 32, 2) # all points during the trajectory

        r = trajectory[:,:,0].astype(np.int32) # (n_future, 32) decompose the points into row
        r = np.clip(r, 0, occ_map.shape[1] - 1)

        c = trajectory[:,:,1].astype(np.int32) # (n_future, 32) decompose the points into column
        c = np.clip(c, 0, occ_map.shape[2] - 1)

        collision = np.full(trajectory.shape[0], False) # we set the length of collision same as the length of trajectory though we only check 4 timesteps
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < occ_map.shape[1]),
                np.logical_and(cc >= 0, cc < occ_map.shape[2]),
            )
            collision[t] = np.any(occ_map[t, rr[I], cc[I]] > OCC_TH)

        return collision

def check_occupancy_for_planned_trajectory_correct(trajectory, data_dict, safe_margin=1., token=None):
    '''
    trajs: torch.Tensor (B, n_future, 2)
    segmentation: torch.Tensor (B, n_future, 200, 200)
    '''
    occ_map = data_dict["occupancy"]

    occ_map = np.fliplr(occ_map.transpose(1,2,0)).transpose(2,0,1)
    occ_map = occ_map[1:] # remove the current timestep
    if occ_map.shape[0] == 4: # if we only have 4 future occupancy
        # New shape
        new_shape = (6, 200, 200)

        # Initialize the new array with the new shape
        expanded_array = np.zeros(new_shape)

        # Copy the original data
        expanded_array[:4] = occ_map

        # Assume that the conditions in the last second continue
        expanded_array[4] = occ_map[-1]
        expanded_array[5] = occ_map[-1]
        occ_map = expanded_array
    
    collision_t = check_collision(CAR_LENGTH+safe_margin, CAR_WIDTH+safe_margin, trajectory, occ_map)

    return collision_t