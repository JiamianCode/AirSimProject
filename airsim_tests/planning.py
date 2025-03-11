import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import distance, ConvexHull
import alphashape
from shapely.geometry import Point, Polygon


matplotlib.use('TkAgg')


def visualize_3d_cloud_with_circle(points_world, center_3d, radius):
    """
    可视化 3D 点云，并绘制检测到的最大内切圆
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    X, Y, Z = points_world[:, 0], points_world[:, 1], points_world[:, 2]
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8)

    # 绘制检测到的圆
    if center_3d is not None and radius > 0:
        # 生成圆的点
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = center_3d[0] + radius * np.cos(theta)
        circle_y = center_3d[1] + radius * np.sin(theta)
        circle_z = np.full_like(circle_x, center_3d[2])  # 保持 z 坐标不变

        ax.plot(circle_x, circle_y, circle_z, color='red', linewidth=2, label="Detected Circle")

        # 标注圆心
        ax.scatter(center_3d[0], center_3d[1], center_3d[2], color='red', s=50, label="Circle Center")

    # 颜色条
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, aspect=10, shrink=0.5)
    cbar.set_label('Z (Elevation)', rotation=15, labelpad=15)

    # 轴标签
    ax.set_xlabel('X', labelpad=12)
    ax.set_ylabel('Y', labelpad=12)
    ax.set_zlabel('Z', labelpad=12)
    ax.view_init(elev=10, azim=-170)

    plt.title("3D Point Cloud with Detected Circle")
    plt.legend()
    plt.tight_layout()
    plt.show()


def minimum_bounding_rectangle(points_2d):
    """
    计算 2D 点云的最小外接矩形 (Minimum Bounding Rectangle, MBR)。
    """
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]

    min_x, min_y = np.min(hull_points, axis=0)
    max_x, max_y = np.max(hull_points, axis=0)

    return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])


def find_best_inscribed_circle(points_2d, alpha=0.2):
    """
    计算点云内部的最佳内切圆：
    - 计算 Alpha Shape（凹多边形），去掉外部异常点
    - 计算最小外接矩形 (MBR)，舍弃细长区域
    - 在剩余点云中找到最佳内切圆
    """
    if len(points_2d) < 3:
        center = np.mean(points_2d, axis=0)
        min_dist = np.min(np.linalg.norm(points_2d - center, axis=1)) if len(points_2d) > 0 else 0
        return center, min_dist

    # 计算 Alpha Shape 形成的凹多边形
    alpha_shape = alphashape.alphashape(points_2d, alpha)

    if not isinstance(alpha_shape, Polygon):
        print("Alpha Shape 计算失败，回退到凸包")
        return np.mean(points_2d, axis=0), np.min(np.linalg.norm(points_2d - np.mean(points_2d, axis=0), axis=1))

    # 计算最小外接矩形，滤除细长部分
    mbr = minimum_bounding_rectangle(points_2d)
    mbr_polygon = Polygon(mbr)

    # 只保留 MBR 内部的点
    filtered_points = np.array([p for p in points_2d if mbr_polygon.contains(Point(p))])

    if len(filtered_points) < 3:
        return np.mean(points_2d, axis=0), np.min(np.linalg.norm(points_2d - np.mean(points_2d, axis=0), axis=1))

    # 计算点云的质心
    centroid = np.mean(filtered_points, axis=0)

    # 选择离质心最近的一个点作为圆心
    min_dist_index = np.argmin(np.linalg.norm(filtered_points - centroid, axis=1))
    best_center = filtered_points[min_dist_index]

    # 计算该圆心到边界的最短距离，作为半径 r
    boundary_points = np.array(alpha_shape.exterior.coords)
    distances = np.array([distance.euclidean(best_center, p) for p in boundary_points])
    r = np.min(distances)

    return best_center, r


def evaluate_planes(planes, drone_position, k, visualize):
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
        center_xy, r = find_best_inscribed_circle(points_2d, alpha=0.2)

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
