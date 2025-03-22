import cv2
import numpy as np
import torch

from airsim_drone.visualization.visualize_3d_cloud_with_circle import visualize_3d_cloud_with_circle


def grid_based_inscribed_circle(points_2d, grid_size=0.05):
    """
    通过栅格化 2D 点云，找到点云的边界，并在其内部找到最大内切圆。

    参数:
    - points_2d: (N, 2) numpy 数组，表示 2D 点云
    - grid_size: float，栅格大小，决定二维网格的精度（默认 0.05 米）

    返回:
    - best_center: (float, float) 圆心坐标
    - max_radius: float 最大内切圆半径
    """
    if len(points_2d) < 3:
        return np.mean(points_2d, axis=0), 0  # 少于 3 个点时，直接返回质心

    # **Step 1: 计算栅格范围**
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)

    grid_width = int((max_x - min_x) / grid_size) + 1
    grid_height = int((max_y - min_y) / grid_size) + 1

    # **Step 2: 构建 occupancy grid (二值图像)**
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for x, y in points_2d:
        grid_x = int((x - min_x) / grid_size)
        grid_y = int((y - min_y) / grid_size)
        occupancy_grid[grid_y, grid_x] = 1  # 反转 y 轴方向

    # **Step 3: 获取边界**
    contours, _ = cv2.findContours(occupancy_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.mean(points_2d, axis=0), 0  # 若无边界，返回质心

    # **Step 4: 计算最大内切圆**
    max_radius = 0
    best_center = np.mean(points_2d, axis=0)  # 先默认质心

    # 遍历整个网格，计算每个内部点到边界的最小距离
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i, j] == 1:  # 仅考虑点云所在区域
                world_x = min_x + j * grid_size
                world_y = min_y + i * grid_size

                # 计算该点到边界的最小距离
                dist = cv2.pointPolygonTest(contours[0], (j, i), True)
                if dist > max_radius:
                    max_radius = dist
                    best_center = (world_x, world_y)

    return best_center, max_radius * grid_size  # 将半径转换回真实坐标


def select_landing_site(planes, drone_position, k, visualize):
    """
    计算无人机与平面相关的评分，并可视化 3D 点云及最佳内切圆。

    参数:
    - planes: List[Dict]，包含多个平面信息（allowed_planes 或 unknown_planes）
    - drone_position: Tuple[float, float, float]，无人机的三维坐标 (x, y, z)
    - k: float，得分计算的权重参数
    - visualize: bool，是否进行 3D 可视化

    返回:
    - sorted_centers_3d: List[Tuple[float, float, float]]，按得分排序的 3D 坐标列表
    - sorted_centers_2d: List[Tuple[int, int]]，按得分排序的 2D 图像坐标列表
    - sorted_scores: List[float]，按得分排序的分数
    """
    results = []

    for plane in planes:
        inlier_points = plane["inlier_points"]
        valid_indices = plane["valid_indices"]  # 2D 图像像素坐标

        # 确保数据为 NumPy 格式
        if isinstance(inlier_points, torch.Tensor):
            inlier_points = inlier_points.cpu().numpy()
        if isinstance(valid_indices, torch.Tensor):
            valid_indices = valid_indices.cpu().numpy()

        # 计算所有点的 xy 坐标 (忽略 Z 轴，令 z=0)
        points_2d = inlier_points[:, :2]

        # 计算最佳内切圆，获取新圆心和半径 r
        center_xy, r = grid_based_inscribed_circle(points_2d, grid_size=0.05)

        center_x, center_y = center_xy

        # 计算圆心的 z 坐标（取平面所有点的均值）
        mean_height = np.mean(inlier_points[:, 2])

        # 计算 h = |无人机高度 - 平面圆心高度|
        drone_x, drone_y, drone_z = drone_position
        h = abs(drone_z - mean_height)  # 确保 h 始终为正

        # 计算无人机到圆心的距离 d
        d = np.linalg.norm(np.array([center_x, center_y, mean_height]) - np.array(drone_position))

        # 计算得分
        if d * h > 0:
            print(f"k:{k}, r:{r:.6f}, d:{d:.6f}, h:{h:.6f}, Score:{(k * (r * r) / (d * h) * 1e6):.2f}")
            score = k * (r * r) / (d * h) * 1e6
        else:
            score = 0

        # 选择最靠近新圆心 (center_x, center_y) 的 3D 点，并找到其对应的 2D 图像坐标
        min_dist_index = np.argmin(np.linalg.norm(inlier_points[:, :2] - center_xy, axis=1))
        closest_2d_point = tuple(valid_indices[min_dist_index])

        results.append(((center_x, center_y, mean_height), closest_2d_point, score))

        if visualize:
            visualize_3d_cloud_with_circle(inlier_points, (center_x, center_y, mean_height), r)

    results.sort(key=lambda x: x[2], reverse=True)

    sorted_centers_3d = [entry[0] for entry in results]
    sorted_centers_2d = [entry[1] for entry in results]
    sorted_scores = [entry[2] for entry in results]

    # 输出结果
    print(sorted_centers_3d, sorted_centers_2d, sorted_scores)

    return sorted_centers_3d, sorted_centers_2d, sorted_scores
