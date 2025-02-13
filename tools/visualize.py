import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def visualize_3d_path(points_world, path_world, start_world, goal_world):
    """
    可视化三维点云，并标注路径、起点和终点。
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    R_world_to_show = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    points_world = (R_world_to_show @ points_world.T).T

    X = points_world[:, 0]
    Y = points_world[:, 1]
    Z = points_world[:, 2]
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8, label='Point Cloud')

    # 绘制路径
    if path_world:
        path_world = np.array(path_world)
        px, py, pz = (R_world_to_show @ path_world.T)
        ax.plot(px, py, pz, c='blue', linewidth=2, label='Path')
    else:
        print("无可行路径，无法可视化路径线段")

    # 绘制起点和终点
    sx, sy, sz = (R_world_to_show @ start_world).T
    gx, gy, gz = (R_world_to_show @ goal_world).T
    ax.scatter(sx, sy, sz, c='green', s=80, marker='o', label='Start')
    ax.scatter(gx, gy, gz, c='red', s=80, marker='x', label='Goal')

    # 颜色条
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, aspect=10, shrink=0.5)
    cbar.set_label('Z (Elevation)', rotation=15, labelpad=15)

    # 轴标签
    ax.set_xlabel('X', labelpad=12)
    ax.set_ylabel('Y', labelpad=12)
    ax.set_zlabel('Z', labelpad=12)
    ax.view_init(elev=10, azim=-170)

    plt.title("3D Point Cloud with Path")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_3d_cloud(points_world):
    """
    可视化三维点云
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    R_world_to_show = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    points_world = (R_world_to_show @ points_world.T).T

    X = points_world[:, 0]
    Y = points_world[:, 1]
    Z = points_world[:, 2]
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8)

    # 颜色条
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, aspect=10, shrink=0.5)
    cbar.set_label('Z (Elevation)', rotation=15, labelpad=15)

    # 轴标签
    ax.set_xlabel('X', labelpad=12)
    ax.set_ylabel('Y', labelpad=12)
    ax.set_zlabel('Z', labelpad=12)
    ax.view_init(elev=10, azim=-170)

    plt.title("3D Point Cloud")
    plt.tight_layout()
    plt.show()
