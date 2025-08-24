import cv2
import numpy as np
import torch

from third_party.DAM.depth_anything_v2.dpt import DepthAnythingV2
from third_party.yoloworld_demo import get_2dloc_open_vocabulary_detector
import matplotlib.pyplot as plt

def pixel_to_3d(depth_map, camera_intrinsic, pixel):
    """
    根据深度图和相机内参，将 2D 像素点转换为 3D 坐标。

    Args:
        depth_map (numpy.ndarray): 深度图。
        camera_intrinsic (numpy.ndarray): 相机内参矩阵，格式为 4x4。
        pixel (tuple): 像素点坐标 (i, j)，其中 i 为行索引 j 为列索引。

    Returns:
        tuple: 3D 坐标 (X, Y, Z)，以米为单位。
    """
    # 获取像素点的深度值
    i, j = pixel
    D = depth_map[j, i]

    def get_average_depth(depth_map, center_pixel, neighborhood_size=5):
        """
        计算给定点周围区域的平均深度值。
        
        :param depth_map: HxW 的原始深度图。
        :param center_pixel: (x, y) 中心点坐标。
        :param neighborhood_size: 周围区域的大小，默认为5x5。
        :return: 平均深度值。
        """
        half_size = neighborhood_size // 2
        x, y = center_pixel
        # 确保不超出边界
        x_start = max(0, x - half_size)
        x_end = min(depth_map.shape[1], x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(depth_map.shape[0], y + half_size + 1)

        neighborhood_depths = depth_map[y_start:y_end, x_start:x_end]
        average_depth = np.mean(neighborhood_depths[neighborhood_depths != 0]) # 忽略零值
        
        return average_depth if not np.isnan(average_depth) else 0
    # breakpoint()
    avg_depth = get_average_depth(depth_map=depth_map, center_pixel=pixel)
    if avg_depth != 0:
        D = avg_depth

    # 从相机内参矩阵中提取焦距和主点坐标
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    u0 = camera_intrinsic[0, 2]
    v0 = camera_intrinsic[1, 2]

    # 计算相机坐标系中的坐标
    X_cam = (i - u0) * D / fx
    Y_cam = (j - v0) * D / fy
    Z_cam = D

    # 返回相机坐标系中的 3D 坐标
    return X_cam, Y_cam, Z_cam

def get_3d_location(text=['car'], image_path=None, debug=False):
    height, width = 900, 1600
    camera_intrinsic = np.array(
        [[1.25281310e+03, 0.00000000e+00, 8.26588115e+02, 0.00000000e+00],
    [0.00000000e+00, 1.25281310e+03, 4.69984663e+02, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],

    )

    model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/depth_anything_v2_vitb.pth'
    # model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/depth_anything_v2_vitl.pth'
    # model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/depth_anything_v2_vits.pth'

    encoder = 'vitb'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder]).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # "location2D": [[1164.0, 627.0]], "location3D": [[1.964, 0.786, 8.525]]
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img) # HxW raw depth map
    # 获取深度图的最大值
    max_depth = np.max(depth)

    # 反转深度图（如果需要）
    depth = max_depth - depth
    if debug:
        plt.imshow(depth, cmap='jet')
        plt.colorbar()
        plt.title("Predicted Depth Map")
        plt.savefig("/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/debug_depth.png")
    prompt = ""
    spatial_location = []
    for obj in text:
        obj = [obj]
        loc2d_prompt, location_2d = get_2dloc_open_vocabulary_detector(text=obj, image_path=image_path)

        prompt += loc2d_prompt
        if location_2d is None:
            prompt += f"The 3d location of {obj} in camera-axis fails to get, you must infer or identify them yourself."
            continue

        pixel = [int(round(coord)) for coord in location_2d]

        # 转换为 3D 坐标
        X, Y, Z = pixel_to_3d(depth, camera_intrinsic, pixel)

        spatial_location.append([X, Y, Z])
        prompt += f"The 3d location of {obj[0]} in camera-axis is {[X, Y, Z]}"
    return prompt, spatial_location
# 示例用法
if __name__ == "__main__":
    # 生成一个示例深度图（单位：米）
    height, width = 900, 1600
    # depth_map = np.random.rand(height, width) * 5.0  # 深度范围为 0 到 5 米

    # 提取相机内参矩阵中的焦距和主点坐标
    # camera_intrinsic = np.array(
    #     [[1.25674851e+03, 0.00000000e+00, 8.17788757e+02, 0.00000000e+00],
    #     [0.00000000e+00, 1.25674851e+03, 4.51954178e+02, 0.00000000e+00], 
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00], 
    #     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
    # )
    camera_intrinsic = np.array(
        [[1.25281310e+03, 0.00000000e+00, 8.26588115e+02, 0.00000000e+00],
    [0.00000000e+00, 1.25281310e+03, 4.69984663e+02, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],

    )


    # model_path = '/mnt/netdata/Team/AI/personal/qiankangan/VLM_ckpt/DEM/Depth-Anything-V2-Small/depth_anything_v2_vits.pth'
    model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/depth_anything_v2_vitb.pth'
    image_path = 'nuscenes_CAM_FRONT_5978.webp'

    encoder = 'vitb'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder]).to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # "location2D": [[1164.0, 627.0]], "location3D": [[1.964, 0.786, 8.525]]
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img) # HxW raw depth map
    print(depth[627, 1164])

    # 指定像素点
    pixel = (1164, 627)  # 行索引为 240，列索引为 320

    # 转换为 3D 坐标
    X, Y, Z = pixel_to_3d(depth, camera_intrinsic, pixel)

    print(f"3D 坐标相机坐标系X = {X:.3f} m, Y = {Y:.3f} m, Z = {Z:.3f} m")