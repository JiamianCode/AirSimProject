import time

import airsim

from main import initialize_depth_model, get_semantic_segmentation, get_depth_estimation, visualize_planes_on_image, \
    setup_cfg
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], 'MaskDINO'))

import cv2
import numpy as np
import torch

from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
# from MaskDINO.demo.predictor import VisualizationDemo
from MaskDINO.demo.mypredictor import MyPredictor
from concurrent.futures import ThreadPoolExecutor, as_completed

import step1
import step2


depth_model_path = 'depth_anything_v2_metric/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
sem_seg_config_file = 'MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml'
sem_seg_model_weights = 'MaskDINO/model/semantic_ade20k_48.7miou.pth'

# 初始化模型
print("初始化开始...")
start_time = time.time()
depth_model = initialize_depth_model(depth_model_path, 'vitl')
predictor, metadata = setup_cfg(sem_seg_config_file, sem_seg_model_weights)
end_time = time.time()
print(f"初始化完成！初始化运行时间: {round(end_time - start_time, 2)}秒")


# 与 airsim 建立连接
client = airsim.MultirotorClient()
client.confirmConnection()

# 确定是否要用API控制
client.enableApiControl(True)

# 解锁无人机转起来
client.armDisarm(True)

# join()等任务结束再进行下个任务
# 起飞
client.takeoffAsync().join()

# 飞行
client.moveToZAsync(-1.5, 1).join()  # 飞到3m高


response = client.simGetImage("front_center", airsim.ImageType.Scene)
f = open('test/in.png', 'wb')
f.write(response)
f.close()

image_path = 'test/in.png'
output_image_path = 'test/out.png'


# 使用并行计算语义分割和深度估计
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(get_semantic_segmentation, image_path, predictor): "sem_seg",
        executor.submit(get_depth_estimation, image_path, depth_model): "depth"
    }

    results = {}
    for future in as_completed(futures):
        task_name = futures[future]
        try:
            results[task_name] = future.result()
        except Exception as e:
            print(f"任务 {task_name} 运行失败: {e}")

# 获取语义分割和深度估计结果
predictions = results.get("sem_seg")
depth = results.get("depth")

# 语义筛选
start_time = time.time()
allowed_masks, unknown_masks = step1.step1(metadata, predictions)
end_time = time.time()
print(f"语义筛选运行时间: {round(end_time - start_time, 2)}秒")

# 平面提取
start_time = time.time()
allowed_planes, unknown_planes = step2.step2(depth, allowed_masks, unknown_masks)
end_time = time.time()
print(f"平面提取运行时间: {round(end_time - start_time, 2)}秒")

# 可视化结果（如果需要）
visualize_planes_on_image(image_path, allowed_planes, unknown_planes, output_image_path)


# 降落
client.landAsync().join()

# 上锁
client.armDisarm(False)
# 释放控制权
client.enableApiControl(False)