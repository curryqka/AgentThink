from ultralytics import YOLOWorld
from ultralytics import YOLO
# from third_party.yoloworld_demo import 

def get_2dbox_open_vocabulary_detector(text=['car'], image_path=None):
    model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/yolov8x-worldv2.pt'

    if not isinstance(text, list):
        text = [text]
    # Initialize a YOLO-World model
    model = YOLO(model_path)  # or choose yolov8m/l-world.pt

    # Define custom classes
    model.set_classes(text)

    # Execute prediction for specified categories on an image
    results = model.predict(image_path)

    box_cls = results[0].boxes.cls
    xywh = results[0].boxes.xywh
    xyxy = results[0].boxes.xyxy

    box_2d = xyxy.cpu().tolist()

    if not box_2d:
        prompt_None = ""
        for id in results[0].names.keys():
            prompt_None += f"\nThe 2d box for {text[id]} in this image fails to get, you must infer or identify them yourself."
        return prompt_None, None
    

    prompt = ""
    for id in results[0].names.keys():
        if id not in box_cls.cpu().tolist():
            prompt += f"\nThe 2d box for {text[id]} in this image fails to get, you must infer or identify them yourself."
            continue
        cur_word_box_id = box_cls.cpu().tolist().index(id)
        prompt += f"\nThe 2d box for {text[id]} in this image is {box_2d[cur_word_box_id]}."
    return prompt, box_2d[0]

def get_2dloc_open_vocabulary_detector(text=['car'], image_path=None):
    model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/yolov8x-worldv2.pt'
    # Initialize a YOLO-World model
    model = YOLO(model_path)  # or choose yolov8m/l-world.pt

    if not isinstance(text, list):
        text = [text]
        
    # Define custom classes
    model.set_classes(text)

    # Execute prediction for specified categories on an image
    results = model.predict(image_path)

    box_cls = results[0].boxes.cls
    xywh = results[0].boxes.xywh
    xyxy = results[0].boxes.xyxy

    # gt: "location2D": [[1164.0, 627.0]], "location3D": [[1.964, 0.786, 8.525]]
    # center_x, center_y = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
   
    pixel_location = xywh.cpu().numpy()[:, 0:2].tolist()

    if not pixel_location:
        prompt_None = ""
        for id in results[0].names.keys():
            prompt_None += f"\nThe 2d location for {text[id]} in this image fails to get, you must infer or identify them yourself."
        return prompt_None, None

    prompt = ""
    for id in results[0].names.keys():
        if id not in box_cls.cpu().tolist():
            prompt += f"\nThe 2d location for {text[id]} in this image fails to get, you must infer or identify them yourself."
            continue
        cur_word_box_id = box_cls.cpu().tolist().index(id)
        
        prompt += f"\nThe 2d location for {text[id]} in this image is {pixel_location[cur_word_box_id]}."
    
    return prompt, pixel_location[0]

if __name__ == '__main__':
    mode = 'diy'
    model_path = '/high_perf_store/mlinfra-vepfs/qiankangan/Drive-MLLM-main/third_party/ckpt/yolov8x-worldv2.pt'
    # image_path = '/high_perf_store/bev_lane/third_party_datasets/nuscenes/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915250362465.jpg'
    image_path = '/high_perf_store/mlinfra-vepfs/llm_data/drive_datasets/MLLM_eval_dataset/validation/image/nuscenes_CAM_FRONT_5976.webp'

    # 
    # text = ['Where is the silver car located in the image?']
    text = ["black motorcycle", 'silver car']
    # text = ["How far apart are black motorcycle and silver car?"]
    if mode == "raw":
        # 加载模型
        model = YOLOWorld(model_path)
        
        # 推理预测
        results = model.predict(data_path)

        
    
    else:
        # Initialize a YOLO-World model
        model = YOLO(model_path)  # or choose yolov8m/l-world.pt
    
        # Define custom classes
        model.set_classes(text)
    
        # Execute prediction for specified categories on an image
        results = model.predict(image_path)

        xywh = results[0].boxes.xywh
        xyxy = results[0].boxes.xyxy

        # gt: "location2D": [[1164.0, 627.0]], "location3D": [[1.964, 0.786, 8.525]]
        # center_x, center_y = (xyxy[0, 0] + xyxy[0, 2]) / 2, (xyxy[0, 1] + xyxy[0, 3]) / 2
        breakpoint()
        # 显示结果
        results[0].show()
        breakpoint()
        # Show results
        results[0].show()