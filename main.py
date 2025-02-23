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


# 深度估计初始化
def initialize_depth_model(model_path, encoder='vitl', max_depth=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置设备：使用 GPU 或 CPU
    # 加载深度估计模型
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(model_path, map_location=device))
    depth_model = depth_model.to(device).eval()
    return depth_model


# 深度估计推理
def get_depth_estimation(image_path, depth_model):
    start_time = time.time()

    raw_image = cv2.imread(image_path)
    # 获取深度图
    depth = depth_model.infer_image(raw_image, 518)  # 默认输入大小

    end_time = time.time()
    print(f"深度估计运行时间: {round(end_time - start_time, 2)}秒")
    return depth


# 语义分割初始化
def setup_cfg(config_file, model_weights):
    # 加载配置和模型
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    # 推理器
    predictor = MyPredictor(cfg)
    # 元数据
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])  # 获取元数据
    return predictor, metadata


# 语义分割推理
def get_semantic_segmentation(image_path, predictor):
    start_time = time.time()

    img = read_image(image_path, format="BGR")
    predictions = predictor.run_on_image(img)  # 获取语义分割结果

    end_time = time.time()
    print(f"语义分割运行时间: {round(end_time - start_time, 2)}秒")
    return predictions


# 可视化平面结果
def visualize_planes_on_image(image_path, allowed_planes, unknown_planes, output_image_path):
    # 读取原始图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    image_output = image_rgb.copy()  # 创建一个副本用于显示叠加结果

    # 定义颜色和透明度（BGR 格式）
    foreground_color = (0, 255, 0)  # 前景色
    background_color = (0, 0, 255)  # 背景色

    # 可视化允许平面（allowed_planes）
    for plane in allowed_planes:
        mask = plane['mask']  # 平面掩码
        category = plane['category']  # 平面类别
        # 创建半透明掩码
        transparent_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        transparent_overlay[mask.cpu().numpy() == 1] = foreground_color
        alpha = 0.5  # 半透明度
        # 将掩码叠加到图像副本上
        image_output = cv2.addWeighted(image_output, 1, transparent_overlay, alpha, 0)

        # 标注类别名称（掩码的中间位置）
        coords = np.column_stack(np.where(mask.cpu().numpy() == 1))
        if len(coords) > 0:
            center = np.mean(coords, axis=0).astype(int)  # 计算掩码区域的中心位置
            center = (center[1], center[0])  # 调整为 (x, y) 格式
            cv2.putText(image_output, category, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 可视化未知平面（unknown_planes）
    for plane in unknown_planes:
        mask = plane['mask']  # 平面掩码
        category = plane['category']  # 平面类别
        # 创建半透明掩码（背景色）
        transparent_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        transparent_overlay[mask.cpu().numpy() == 1] = background_color
        alpha = 0.5  # 透明度
        # 将掩码叠加到图像副本上
        image_output = cv2.addWeighted(image_output, 1, transparent_overlay, alpha, 0)

        # 标注类别名称（掩码的中间位置）
        coords = np.column_stack(np.where(mask.cpu().numpy() == 1))
        if len(coords) > 0:
            center = np.mean(coords, axis=0).astype(int)  # 计算掩码区域的中心位置
            center = (center[1], center[0])  # 调整为 (x, y) 格式
            cv2.putText(image_output, category, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 保存图像
    cv2.imwrite(output_image_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))  # 转回 BGR 格式保存

    # 创建可调整大小的窗口
    # cv2.namedWindow("Visualized Planes", cv2.WINDOW_NORMAL)  # 设置窗口为可调大小
    # cv2.imshow("Visualized Planes", cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))  # 转回 BGR 格式显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# 主程序
if __name__ == "__main__":
    # 模型路径和参数
    name = '12'
    image_path = 'assets/UAV/images/' + name + '.jpg'
    output_image_path = 'assets/UAV/output/' + name + '.png'
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
