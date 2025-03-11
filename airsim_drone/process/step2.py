import torch
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import label as cpu_label
from scipy.spatial import KDTree
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Dict

from examples.Astar.visualize import visualize_3d_cloud


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
        title = f"{plane_type} Plane {i + 1}"
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

def gaussian_smoothing(points, k_neighbors, sigma):
    """
    对 3D 点云进行高斯平滑，使平面更加平坦，减少噪声。

    参数：
    - points: (N, 3) numpy 数组，表示点云数据
    - k_neighbors: 每个点考虑的近邻个数
    - sigma: 高斯核的标准差，决定平滑强度

    返回：
    - smoothed_points: (N, 3) numpy 数组，平滑后的点云
    """
    points_np = np.asarray(points)
    tree = KDTree(points_np)  # 构建 KDTree 加速邻域搜索
    smoothed_points = np.zeros_like(points_np)

    for i, point in enumerate(points_np):
        # 查找 K 近邻（包括自身）
        distances, indices = tree.query(point, k=k_neighbors)

        # 计算高斯权重
        weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))
        weights /= np.sum(weights)  # 归一化

        # 计算加权均值
        smoothed_points[i] = np.sum(points_np[indices] * weights[:, None], axis=0)

    return smoothed_points

def detect_horizontal_planes(points, filter_config):
    """从点云中提取多个水平平面及其信息，并基于密度阈值筛选有效平面"""
    # 提取参数
    distance_threshold = filter_config.get('distance_threshold')
    max_iterations = filter_config.get('max_iterations')
    min_points = filter_config.get('min_points')

    angle_threshold = filter_config.get('angle_threshold')
    max_planes = filter_config.get('max_planes')

    # 输入类型处理
    is_torch = isinstance(points, torch.Tensor)
    device = points.device if is_torch else 'cpu'
    points_np = points.cpu().numpy() if is_torch else np.asarray(points)

    # 高斯平滑 (废弃)
    # points_np = gaussian_smoothing(points_np, k_neighbors=200, sigma=5)

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
            plane_model, local_inliers = pcd.segment_plane(distance_threshold, min_points, max_iterations)
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
        })

        # 从剩余点中移除当前内点
        remaining_indices = np.delete(remaining_indices, local_inliers)

    return planes


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

    # 掩码侵蚀操作
    mask_cpu = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask  # 转换为 NumPy 数组
    erosion_kernel_size = filter_config.get("erosion_kernel_size")  # 侵蚀核大小，默认为 3
    erosion_iterations = filter_config.get("erosion_iterations")  # 侵蚀次数

    # 进行多次侵蚀操作
    for _ in range(erosion_iterations):
        mask_cpu = binary_erosion(mask_cpu, structure=np.ones((erosion_kernel_size, erosion_kernel_size)))

    # 计算连通区域
    labeled_mask, num_labels = cpu_label(mask_cpu)
    # 计算区域大小
    label_sizes = {label: np.sum(labeled_mask == label) for label in range(1, num_labels + 1)}
    # 筛选出大于 min_area的区域
    min_area = filter_config.get("min_area")
    valid_labels = [label for label, size in label_sizes.items() if size >= min_area]
    if not valid_labels:
        return planes  # 没有有效区域，直接返回空列表
    labeled_mask = torch.from_numpy(labeled_mask).to(device)

    # 提取有效像素点 的 连通区域ID
    u = valid_indices[:, 0].long()
    v = valid_indices[:, 1].long()
    region_labels = labeled_mask[v, u]  # 有效像素位置 的 连通区域 编号， 大小（N，）

    # 处理每个区域
    for label_id in range(1, num_labels + 1):
        mask_per_label = (region_labels == label_id)  # 当前连通区域 的 有效像素位置 掩码， 大小（N，）
        region_points = points[mask_per_label]  # 像素对应点云，大小（N，3）

        if len(region_points) < filter_config.get('min_points'):  # 跳过点数不足的区域
            continue

        # 检测多个水平平面
        try:
            detected_planes = detect_horizontal_planes(region_points, filter_config)
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

            # 计算区域密度
            mask_indices = torch.nonzero(current_mask, as_tuple=False)
            if len(mask_indices) > 0:
                ymin, xmin = mask_indices.min(dim=0).values  # 计算边界框最小值
                ymax, xmax = mask_indices.max(dim=0).values  # 计算边界框最大值

                bbox_area = (ymax - ymin + 1) * (xmax - xmin + 1)
                mask_area = current_mask.sum().item()  # 实际占据的像素数

                density = mask_area / bbox_area if bbox_area > 0 else 0
            else:
                density = 0

            # 如果密度低于阈值，则剔除
            if density < filter_config.get('density_threshold', 0.5):
                continue

            plane_info["category"] = category
            plane_info["mask"] = current_mask
            plane_info["density"] = density
            plane_info["valid_indices"] = valid_indices[mask_per_label][inlier_mask]

            # visualize_3d_cloud(plane_info["inlier_points"].cpu().numpy())

            planes.append(plane_info)

    return planes


def preprocess_planes(voxel_manager, allowed_planes, unknown_planes, filter_config):
    """
    预处理平面数据
    """
    check_height = filter_config.get('check_height')
    min_area = filter_config.get('min_area')

    filtered_allowed = []
    filtered_unknown = []

    global_voxel_grid = voxel_manager.global_voxel_grid
    voxel_size = global_voxel_grid.voxel_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def filter_plane(plane):
        new_planes = []

        points = plane["inlier_points"]
        mask = plane["mask"].clone()
        valid_indices = plane["valid_indices"]

        if len(points) == 0:
            return None

        is_tensor = isinstance(points, torch.Tensor)
        if is_tensor:
            points_np = points.cpu().numpy()
            valid_indices_np = valid_indices.cpu().numpy()
        else:
            points_np = np.array(points)
            valid_indices_np = np.array(valid_indices)

        voxel_keys = global_voxel_grid.get_voxel_keys_batch(points_np)

        # 在check_height高度范围内检查是否有障碍物
        keep_indices = []
        for idx, voxel in enumerate(voxel_keys):
            x, y, z = voxel
            safe = True

            steps = int(check_height / voxel_size)
            for dz in range(1, steps + 1):
                check_voxel = (x, y, z - dz)
                if check_voxel in global_voxel_grid.grid:
                    safe = False
                    break

            if safe:
                keep_indices.append(idx)

        if keep_indices:
            # 只保留未受影响的点
            filtered_points_np = points_np[keep_indices]
            filtered_valid_indices = valid_indices_np[keep_indices]

            # 更新 mask
            mask_np = mask.cpu().numpy() if is_tensor else mask
            current_mask = np.zeros_like(mask_np)  # **重新初始化掩码**
            u, v = filtered_valid_indices[:, 0], filtered_valid_indices[:, 1]
            current_mask[v, u] = 1


            # 计算掩码的连通区域
            labeled_mask, num_labels = cpu_label(current_mask)
            # 计算每个区域的大小
            label_sizes = {label: np.sum(labeled_mask == label) for label in range(1, num_labels + 1)}
            # 仅保留面积 >= `min_area` 的区域
            valid_labels = [label for label, size in label_sizes.items() if size >= min_area]
            if not valid_labels:
                return None  # 若无符合区域，直接丢弃

            for label_id in valid_labels:
                filtered_mask = (labeled_mask == label_id).astype(np.uint8)

                # 提取该连通区域的像素索引
                mask_indices = np.argwhere(filtered_mask > 0)
                new_valid_indices = np.column_stack((mask_indices[:, 1], mask_indices[:, 0]))  # (u, v)

                # 仅保留符合区域的点云
                valid_mask = np.isin(labeled_mask[v, u], [label_id])
                final_points_np = filtered_points_np[valid_mask]

                # 若最终点数过少，丢弃该平面
                if len(final_points_np) < min_area:
                    continue

                # 计算密度
                if len(mask_indices) > 0:
                    ymin, xmin = np.min(mask_indices, axis=0)  # 计算边界框最小值
                    ymax, xmax = np.max(mask_indices, axis=0)  # 计算边界框最大值

                    bbox_area = (ymax - ymin + 1) * (xmax - xmin + 1)
                    mask_area = filtered_mask.sum()  # 计算实际占据的像素数

                    density = mask_area / bbox_area if bbox_area > 0 else 0
                else:
                    density = 0
                # 如果密度低于阈值，则剔除
                if density < filter_config.get('density_threshold'):
                    continue

                # 重新构建平面信息
                new_plane = plane.copy()
                new_plane["inlier_points"] = torch.from_numpy(final_points_np).to(device) if is_tensor else final_points_np
                new_plane["valid_indices"] = torch.tensor(new_valid_indices, device=device) if is_tensor else new_valid_indices
                new_plane["mask"] = torch.tensor(filtered_mask, device=device) if is_tensor else filtered_mask
                new_plane["density"] = density
                '''
                plt.figure()
                plt.title(f"plane")
                plt.imshow(filtered_mask, cmap="gray")
                plt.axis("off")
                plt.show()
                '''
                new_planes.append(new_plane)

        return new_planes

    for plane in allowed_planes:
        new_planes = filter_plane(plane)
        if new_planes:
            filtered_allowed.extend(new_planes)

    for plane in unknown_planes:
        new_planes = filter_plane(plane)
        if new_planes:
            filtered_unknown.extend(new_planes)

    return filtered_allowed, filtered_unknown


def step2(voxel_manager, points, valid_indices, allowed_masks, unknown_masks, filter_config):
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

    # 处理allowed_masks
    for category, mask in allowed_masks.items():
        planes = process_mask(mask, valid_indices, points, category, filter_config)
        allowed_planes.extend(planes)

    # 处理unknown_masks
    for category, mask in unknown_masks.items():
        planes = process_mask(mask, valid_indices, points, category, filter_config)
        unknown_planes.extend(planes)

    allowed_planes, unknown_planes = preprocess_planes(voxel_manager, allowed_planes, unknown_planes, filter_config)

    # 可视化平面
    # show_all_planes(allowed_planes, unknown_planes)

    return allowed_planes, unknown_planes
