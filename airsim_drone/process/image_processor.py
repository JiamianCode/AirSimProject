import cv2
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

from airsim_drone.process import step1, step2


class ImageProcessor:
    def __init__(self, depth_model_path, sem_seg_config_file, sem_seg_model_weights):
        """
        初始化深度估计模型和语义分割模型
        """
        print("ImageProcessor 初始化开始...")
        start_time = time.time()

        # 加载深度估计模型
        self.depth_model = self.initialize_depth_model(depth_model_path, 'vitl')
        # 加载语义分割模型
        self.predictor, self.metadata = self.setup_cfg(sem_seg_config_file, sem_seg_model_weights)

        end_time = time.time()
        print(f"ImageProcessor 初始化完成！初始化运行时间: {round(end_time - start_time, 2)}秒")

    def process_image(self, image):
        """
        处理图像，进行语义分割和深度估计

        :param image: 输入图像 (numpy.ndarray 格式)
        :return: 语义分割结果、深度估计结果
        """
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
        allowed_planes, unknown_planes = step2.step2(voxel_manager, points, valid_indices, allowed_masks, unknown_masks, filter_config)
        end_time = time.time()
        print(f"平面提取完成，运行时间: {round(end_time - start_time, 2)}秒")

        return allowed_planes, unknown_planes

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

    @staticmethod
    def sharpen_real_depth(depth, threshold, dilate_size):
        """
        彻底割裂真实深度图的前后景边界，确保点云生成正确。

        参数：
        - depth_map: 输入深度图 (numpy 数组，单位为毫米)
        - threshold: 梯度阈值，高于此值的地方认为是前后景交界(单位mm)
        - dilate_size: 形态学膨胀核的大小，用于增强边界

        返回：
        - 处理后的深度图
        """
        # 计算梯度 (Sobel 过滤器, 单位仍然是深度)
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 生成边缘掩码 (深度变化剧烈的区域)
        edge_mask = gradient_magnitude > threshold
        edge_mask = edge_mask.astype(np.uint8) * 255  # 转换为二值图

        # 形态学膨胀，扩大边缘区域
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        # 计算前景和背景的深度均值
        foreground_depth = cv2.blur(depth, (5, 5))  # 平均深度
        background_depth = cv2.medianBlur(depth, 5)  # 消除噪声

        # 找到边缘附近的前景和背景
        edge_indices = np.where(edge_mask > 0)
        depth_foreground_vals = foreground_depth[edge_indices]
        depth_background_vals = background_depth[edge_indices]

        # 确保前景背景不会混合，直接拉回前后景均值
        adjusted_depth = depth.copy()
        adjusted_depth[edge_indices] = np.where(
            depth[edge_indices] < np.median(depth),
            np.median(depth_foreground_vals),
            np.median(depth_background_vals)
        )

        return adjusted_depth

    @staticmethod
    def visualize_planes_on_image(image, allowed_planes, unknown_planes, sorted_centers_2d=None, sorted_scores=None):
        """
        可视化平面掩码，并在原始图像上叠加类别标签。
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        image_output = image_rgb.copy()  # 创建一个副本用于显示叠加结果

        # 定义颜色和透明度（BGR 格式）
        allowed_color = (0, 255, 0)    # 绿色 (允许平面)
        unknown_color = (0, 0, 255)    # 红色 (未知平面)
        alpha = 0.5                    # 半透明度

        # 绘制允许的平面
        for plane in allowed_planes:
            mask = plane['mask']  # 平面掩码
            category = plane['category']  # 平面类别
            # 创建半透明掩码
            transparent_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
            transparent_overlay[mask.cpu().numpy() == 1] = allowed_color
            # 将掩码叠加到图像副本上
            image_output = cv2.addWeighted(image_output, 1, transparent_overlay, alpha, 0)

            # 标注类别名称（掩码的中间位置）
            coords = np.column_stack(np.where(mask.cpu().numpy() == 1))
            if len(coords) > 0:
                center = np.mean(coords, axis=0).astype(int)  # 计算掩码区域的中心位置
                center = (center[1], center[0])  # 调整为 (x, y) 格式
                cv2.putText(image_output, category, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # 绘制未知的平面
        for plane in unknown_planes:
            mask = plane['mask']  # 平面掩码
            category = plane['category']  # 平面类别
            # 创建半透明掩码（背景色）
            transparent_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
            transparent_overlay[mask.cpu().numpy() == 1] = unknown_color
            # 将掩码叠加到图像副本上
            image_output = cv2.addWeighted(image_output, 1, transparent_overlay, alpha, 0)

            # 计算掩码区域的中心位置并添加文本
            coords = np.column_stack(np.where(mask.cpu().numpy() == 1))
            if len(coords) > 0:
                center = np.mean(coords, axis=0).astype(int)  # 计算掩码区域的中心位置
                center = (center[1], center[0])  # 调整为 (x, y) 格式
                cv2.putText(image_output, category, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 绘制 sorted_centers_2d
        if sorted_centers_2d:
            num_centers = len(sorted_centers_2d)
            for i, (u, v) in enumerate(sorted_centers_2d):
                red_intensity = 255 - int((i / max(1, num_centers - 1)) * 180)  # 颜色渐变
                center_color = (0, 0, red_intensity)

                # 画圆点
                cv2.circle(image_output, (u, v), 5, center_color, -1)

                # 标注编号 (1, 2, 3...)
                cv2.putText(image_output, f"{i+1}", (u+10, v+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # 标注得分
                if sorted_scores:
                    cv2.putText(image_output, f"{sorted_scores[i]:.2f}", (u+10, v+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 创建可调整大小的窗口并显示结果
        cv2.namedWindow("Visualized Planes", cv2.WINDOW_NORMAL)  # 设置窗口为可调大小
        cv2.imshow("Visualized Planes", cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))  # 转回 BGR 格式显示
        cv2.waitKey(0)
        cv2.destroyAllWindows()
