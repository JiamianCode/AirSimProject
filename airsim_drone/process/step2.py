import torch
import numpy as np
from scipy.ndimage import label as cpu_label
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Dict


def visualize_planes(planes: List[Dict], plane_type: str = "Allowed"):
    """逐个展示平面可视化，显示详细信息"""
    if not planes:
        print(f"No {plane_type} planes to visualize")
        return

    for i, plane in enumerate(planes):
        plt.figure(figsize=(6, 6))  # 每次只展示一张图

        # 处理数据
        mask = plane["mask"].cpu().numpy() if isinstance(plane["mask"], torch.Tensor) else plane["mask"]
        category = plane.get("category", "Unknown")
        normal = plane["normal"].cpu().numpy() if isinstance(plane["normal"], torch.Tensor) else plane["normal"]
        offset = plane["offset"].item() if isinstance(plane["offset"], torch.Tensor) else plane["offset"]
        inliers = plane["inlier_points"].shape[0] if "inlier_points" in plane else 0
        tilt_angle = plane["angle"]
        density = plane.get("density", "N/A")

        # 生成标题文本
        title = f"{plane_type} Plane {i+1}"
        info_text = (
            f"Category: {category}\n"
            f"Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]\n"
            f"Offset: {offset:.2f}\n"
            f"Inliers: {inliers}\n"
            f"Tilt: {tilt_angle:.1f}°\n"
            f"Density: {density:.2f}"
        )

        # 可视化 mask
        plt.imshow(mask, cmap="gray")
        plt.title(title, fontsize=14)
        plt.axis("off")

        # 在图像上添加文本信息
        text_x, text_y = mask.shape[1] // 20, mask.shape[0] // 10  # 设定文本位置
        plt.text(
            text_x, text_y, info_text, color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.5")
        )

        # 显示并暂停，等待用户关闭窗口
        plt.show()


def show_all_planes(allowed_planes, unknown_planes):
    """展示所有检测到的平面"""
    # 显示允许的平面
    visualize_planes(allowed_planes, plane_type="Allowed")

    # 显示未知平面
    visualize_planes(unknown_planes, plane_type="Unknown")


def detect_horizontal_planes(points, filter_config):
    """从点云中提取多个水平平面及其信息，并基于密度阈值筛选有效平面"""

    # 提取参数
    distance_threshold = filter_config.get('distance_threshold', 0.02)
    max_iterations = filter_config.get('max_iterations', 1000)
    min_points = filter_config.get('min_points', 3)

    angle_threshold = filter_config.get('angle_threshold', 5)
    density_threshold = filter_config.get('density_threshold', 0.8)
    max_planes = filter_config.get('max_planes', 5)

    # 输入类型处理
    is_torch = isinstance(points, torch.Tensor)
    device = points.device if is_torch else 'cpu'
    points_np = points.cpu().numpy() if is_torch else np.asarray(points)

    remaining_indices = np.arange(len(points_np))  # 记录剩余点的全局索引
    planes = []

    for _ in range(max_planes):
        if len(remaining_indices) < min_points:
            break

        # 创建当前剩余点的 Open3D 点云
        current_points = points_np[remaining_indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(current_points)

        try:
            # 使用 RANSAC 分割平面 (返回当前剩余点云中的内点索引)
            plane_model, local_inliers = pcd.segment_plane(
                distance_threshold,
                min_points,
                max_iterations
            )
        except RuntimeError:
            break

        # 解析平面参数
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal_norm = normal / np.linalg.norm(normal)  # 归一化法向量

        # 计算法向角度
        vertical_axis = np.array([0, 0, 1])
        dot_product = np.abs(np.dot(normal_norm, vertical_axis))
        angle = np.degrees(np.arccos(dot_product))  # 计算夹角（度）

        if angle > angle_threshold:
            # 如果平面不水平，则跳过并移除内点
            remaining_indices = np.delete(remaining_indices, local_inliers)
            continue

        # 记录全局内点索引
        global_inliers = remaining_indices[local_inliers]
        inlier_mask = np.zeros(len(points_np), dtype=bool)
        inlier_mask[global_inliers] = True

        # 计算区域密度
        indices = np.where(inlier_mask)[0]  # 仅获取 `True` 索引
        if len(indices) > 0:
            min_idx = indices.min()  # 计算最小索引
            max_idx = indices.max()  # 计算最大索引

            bbox_length = max_idx - min_idx + 1  # 计算边界范围
            bbox_area = bbox_length * distance_threshold**2  # 估算区域面积
            max_possible_points = bbox_area / (distance_threshold ** 2)  # 计算最大可能点数
            density = inlier_mask.sum() / max_possible_points if max_possible_points > 0 else 0
        else:
            density = 0

        if density < density_threshold:
            # 如果密度太低，则跳过
            remaining_indices = np.delete(remaining_indices, local_inliers)
            continue

        # 生成 PyTorch 掩码（如果输入是张量）
        if is_torch:
            normal = torch.from_numpy(normal).to(device).float()
            inlier_mask = torch.tensor(inlier_mask, device=device)
        inlier_points = points[global_inliers] if is_torch else points_np[global_inliers]
        # 保存平面信息
        planes.append({
            "normal": normal,
            "offset": torch.tensor(d).to(device) if is_torch else d,
            "inlier_mask": inlier_mask,
            "inlier_points": inlier_points,
            "angle": angle,
            "density": density
        })

        # 从剩余点中移除当前内点
        remaining_indices = np.delete(remaining_indices, local_inliers)

    return planes


def remove_small_connected_components_gpu(mask, min_area):
    """GPU加速的小连通区域剔除"""
    mask_cpu = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    labeled_mask, num_labels = cpu_label(mask_cpu)

    cleaned_mask_cpu = np.zeros_like(mask_cpu)
    for label in range(1, num_labels + 1):
        if np.sum(labeled_mask == label) >= min_area:
            cleaned_mask_cpu[labeled_mask == label] = 1

    return torch.tensor(cleaned_mask_cpu, device=mask.device) if isinstance(mask, torch.Tensor) else cleaned_mask_cpu


def process_mask(mask, valid_indices, points, category, filter_config):
    """处理掩码并返回多个平面信息"""
    # 统一设备并确保张量输入
    device = mask.device if isinstance(mask, torch.Tensor) else 'cpu'

    # 转换输入为张量
    if not isinstance(valid_indices, torch.Tensor):
        valid_indices = torch.from_numpy(valid_indices).to(device)
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device)

    planes = []
    mask_cpu = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    labeled_mask_np, num_labels = cpu_label(mask_cpu)
    labeled_mask = torch.from_numpy(labeled_mask_np).to(device)

    # 提取区域标签
    u = valid_indices[:, 0].long()
    v = valid_indices[:, 1].long()
    region_labels = labeled_mask[v, u]

    # 处理每个区域
    for label_id in range(1, num_labels + 1):
        mask_per_label = (region_labels == label_id)
        region_points = points[mask_per_label]

        if len(region_points) < 3:  # 跳过点数不足的区域
            continue

        # 检测多个水平平面
        try:
            detected_planes = detect_horizontal_planes(
                region_points,
                filter_config
            )
        except Exception as e:
            print(f"平面检测失败: {str(e)}")
            continue

        # 处理每个检测到的平面
        for plane_info in detected_planes:
            # 处理内点掩码
            inlier_mask = plane_info['inlier_mask']

            # 构建掩码映射回原图
            current_mask = torch.zeros_like(mask)
            current_mask[v[mask_per_label][inlier_mask], u[mask_per_label][inlier_mask]] = 1

            plane_info["category"] = category
            plane_info["mask"] = current_mask

            planes.append(plane_info)

        return planes


def step2(points, valid_indices, allowed_masks, unknown_masks, filter_config):
    """主流程中统一数据类型处理"""
    # 将输入转换为张量并送至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device)
    if not isinstance(valid_indices, torch.Tensor):
        valid_indices = torch.from_numpy(valid_indices).to(device)

    # 处理掩码时确保数据类型一致
    allowed_masks = {k: v.to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device) for k, v in
                     allowed_masks.items()}
    unknown_masks = {k: v.to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device) for k, v in
                     unknown_masks.items()}

    allowed_planes = []
    unknown_planes = []

    for category, mask in allowed_masks.items():
        cleaned_mask = remove_small_connected_components_gpu(mask, min_area=1000)
        planes = process_mask(cleaned_mask, valid_indices, points, category, filter_config)
        allowed_planes.extend(planes)

    for category, mask in unknown_masks.items():
        cleaned_mask = remove_small_connected_components_gpu(mask, min_area=1000)
        planes = process_mask(cleaned_mask, valid_indices, points, category, filter_config)
        unknown_planes.extend(planes)

    # 可视化平面
    show_all_planes(allowed_planes, unknown_planes)

    return allowed_planes, unknown_planes
