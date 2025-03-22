import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def visualize_global_voxel_grid(voxel_manager, remove_top=True):
    """可视化全局体素网格，支持去顶操作"""
    matplotlib.use('TkAgg')

    occupied_indices = np.array(list(voxel_manager.global_voxel_grid.grid.keys()))
    if occupied_indices.shape[0] == 0:
        print("无可视化数据：体素网格为空")
        return

    # 默认体素颜色为灰色
    colors = np.full((occupied_indices.shape[0], 3), [128, 128, 128], dtype=np.uint8)

    # 如果 remove_top 为 True，进行去顶操作
    if remove_top:
        z_values = occupied_indices[:, 2] * voxel_manager.global_voxel_grid.voxel_size + voxel_manager.global_origin[2]
        z_min = np.min(z_values)
        z_max = np.max(z_values)
        z_min_cutoff = z_max - 0.75 * (z_max - z_min)  # 保留 75% 的高度

        # 只保留在此范围内的点
        valid_indices = z_values >= z_min_cutoff
        occupied_indices = occupied_indices[valid_indices]
        colors = colors[valid_indices]  # 颜色数组同步调整

    # 提取 X, Y, Z 坐标
    X = occupied_indices[:, 0] * voxel_manager.global_voxel_grid.voxel_size + voxel_manager.global_origin[0]
    Y = occupied_indices[:, 1] * voxel_manager.global_voxel_grid.voxel_size + voxel_manager.global_origin[1]
    Z = occupied_indices[:, 2] * voxel_manager.global_voxel_grid.voxel_size + voxel_manager.global_origin[2]

    # 可视化体素网格
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8)

    # 设置轴标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Voxel Grid")
    ax.view_init(elev=190, azim=-20)

    plt.show()
