import os

import torch
import numpy as np
import time

from depth_anything_v2_metric.depth_anything_v2.dpt import DepthAnythingV2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from MaskDINO.maskdino import add_maskdino_config
# from MaskDINO.demo.predictor import VisualizationDemo
from MaskDINO.demo.mypredictor import MyPredictor
from concurrent.futures import ThreadPoolExecutor, as_completed

from airsim_drone.process import step1, step2, step3


class ImageProcessor:
    depth_model_path = '../../depth_anything_v2_metric/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
    sem_seg_config_file = '../../MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml'
    sem_seg_model_weights = '../../MaskDINO/model/semantic_ade20k_48.7miou.pth'

    def __init__(self):
        """
        初始化深度估计模型和语义分割模型
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.depth_model_path = os.path.abspath(os.path.join(script_dir, self.depth_model_path))
        self.sem_seg_config_file = os.path.abspath(os.path.join(script_dir, self.sem_seg_config_file))
        self.sem_seg_model_weights = os.path.abspath(os.path.join(script_dir, self.sem_seg_model_weights))

        print("ImageProcessor 初始化开始...")
        start_time = time.time()

        # 加载深度估计模型
        self.depth_model = self.initialize_depth_model(self.depth_model_path, 'vitl')
        # 加载语义分割模型
        self.predictor, self.metadata = self.setup_cfg(self.sem_seg_config_file, self.sem_seg_model_weights)

        end_time = time.time()
        print(f"ImageProcessor 初始化完成！初始化运行时间: {round(end_time - start_time, 2)}秒")

    # 深度估计初始化
    @staticmethod
    def initialize_depth_model(model_path, encoder='vitl', max_depth=20):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择 GPU 或 CPU
        # 加载深度估计模型
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        depth_model.load_state_dict(torch.load(model_path, map_location=device))
        depth_model = depth_model.to(device).eval()
        return depth_model

    # 深度估计推理（接受 numpy 图像）
    @staticmethod
    def get_depth_estimation(image, depth_model):
        start_time = time.time()

        if not isinstance(image, np.ndarray):
            raise ValueError("输入图像必须是 numpy.ndarray 类型")

        # 获取深度图
        depth = depth_model.infer_image(image, 518)  # 默认输入大小

        end_time = time.time()
        print(f"深度估计运行时间: {round(end_time - start_time, 2)}秒")
        return depth

    # 语义分割初始化
    @staticmethod
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

    # 语义分割推理（接受 numpy 图像）
    @staticmethod
    def get_semantic_segmentation(image, predictor):
        start_time = time.time()

        if not isinstance(image, np.ndarray):
            raise ValueError("输入图像必须是 numpy.ndarray 类型")

        predictions = predictor.run_on_image(image)  # 获取语义分割结果

        end_time = time.time()
        print(f"语义分割运行时间: {round(end_time - start_time, 2)}秒")
        return predictions

    def process_image(self, image):
        """处理图像，进行语义分割和深度估计"""
        print('开始语义分割与深度估计...')
        start_time = time.time()

        # 使用线程池并行计算语义分割和深度估计
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.get_semantic_segmentation, image, self.predictor): "sem_seg",
                executor.submit(self.get_depth_estimation, image, self.depth_model): "depth"
            }

            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    print(f"任务 {task_name} 运行失败: {e}")

        end_time = time.time()
        print(f"语义分割与深度估计完成，运行时间: {round(end_time - start_time, 2)}秒")

        predictions, depth = results.get("sem_seg"), results.get("depth")

        return predictions, depth

    def filter_and_extract_planes(self, voxel_manager, predictions, points, valid_indices):
        """
        对语义分割结果进行筛选，并基于深度信息提取平面
        """
        # 语义筛选
        print("开始语义筛选...")
        start_time = time.time()
        allowed_masks, unknown_masks = step1.step1(self.metadata, predictions, min_area=1000)
        end_time = time.time()
        print(f"语义筛选完成，运行时间: {round(end_time - start_time, 2)}秒")

        # 平面提取
        print("开始平面提取...")
        start_time = time.time()
        # 配置过滤参数
        filter_config = {
            "min_area": 1000,  # 连通区域的最小面积
            "erosion_kernel_size": 5,  # 侵蚀核大小，默认为 3
            "erosion_iterations": 3,  # 侵蚀次数

            "check_height": 0.5,  # 平面上方区域检查的高度

            "distance_threshold": 0.03,  # 平面拟合的距离阈值
            "max_iterations": 5000,  # RANSAC 最大迭代次数
            "min_points": 50,  # 最小点数要求

            "angle_threshold": 10,  # 角度阈值 (度数)
            "max_planes": 5,  # 允许检测的最大平面数量
            "density_threshold": 0.3  # 区域密度阈值
        }
        allowed_planes, unknown_planes = step2.step2(voxel_manager, points, valid_indices, allowed_masks, unknown_masks,
                                                     filter_config)
        end_time = time.time()
        print(f"平面提取完成，运行时间: {round(end_time - start_time, 2)}秒")

        extracted_planes = step3.step3(predictions, self.metadata, points, valid_indices, filter_config)

        return allowed_planes, unknown_planes, extracted_planes
