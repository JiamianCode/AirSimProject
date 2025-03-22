import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def visualize_voxel_grid_with_path(voxel_manager, path, remove_top=True):
    """可视化体素网格（障碍物）和 路径，并可选择去顶"""
    matplotlib.use('TkAgg')

    # 获取占用体素索引
    occupied_indices = np.array(list(voxel_manager.global_voxel_grid.grid.keys()))
    if occupied_indices.shape[0] == 0:
        print("无可视化数据：体素网格为空")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 计算体素坐标
    voxel_size = voxel_manager.global_voxel_grid.voxel_size
    global_origin = voxel_manager.global_origin

    X = occupied_indices[:, 0] * voxel_size + global_origin[0]
    Y = occupied_indices[:, 1] * voxel_size + global_origin[1]
    Z = occupied_indices[:, 2] * voxel_size + global_origin[2]

    # 如果启用去顶操作，去除顶部体素
    if remove_top:
        z_min = np.min(Z)
        z_max = np.max(Z)
        z_min_cutoff = z_max - 0.75 * (z_max - z_min)  # 仅保留 75% 低处的点
        valid_indices = Z >= z_min_cutoff

        X, Y, Z = X[valid_indices], Y[valid_indices], Z[valid_indices]

    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8, label='Voxel Grid')

    # 绘制路径
    if path is not None and len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], c='blue', linewidth=2, label='Planned Path')
    else:
        print("无可行路径，无法可视化路径线段")

    # 标记起点和终点
    start, goal = path[0], path[-1]
    ax.scatter(start[0], start[1], start[2], c='green', s=80, marker='o', label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='red', s=80, marker='x', label='Goal')

    # 轴标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Voxel Grid with Path")
    ax.view_init(elev=190, azim=-20)
    plt.legend()
    plt.show()
