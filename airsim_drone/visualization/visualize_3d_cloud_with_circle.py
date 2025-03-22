import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def visualize_3d_cloud_with_circle(points_world, center_3d, radius):
    """
    可视化 3D 点云，并绘制检测到的最大内切圆
    """
    matplotlib.use('TkAgg')

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
