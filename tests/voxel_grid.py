from airsim_drone import AirSimDroneController
from airsim_drone.visualization.visualize import visualize_3d_cloud



def create_voxel_grid(points_world, resolution=0.1, margin=0.2):
    """
    将点云数据转换为 Voxel Grid（体素网格）
    :param points_world: (N,3) 形状的 numpy 数组，表示世界坐标点云
    :param resolution: 体素网格的大小，默认为 0.1m
    :param margin: 额外的安全边界
    :return: occupancy_grid, origin, shape
    """
    if len(points_world) == 0:
        raise ValueError("点云为空，无法构建体素网格")

    # 获取点云边界
    x_min, x_max = np.min(points_world[:, 0]), np.max(points_world[:, 0])
    y_min, y_max = np.min(points_world[:, 1]), np.max(points_world[:, 1])
    z_min, z_max = np.min(points_world[:, 2]), np.max(points_world[:, 2])

    # 增加安全边界
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    z_min -= margin
    z_max += margin

    # 计算网格大小
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    nz = int(np.ceil((z_max - z_min) / resolution))

    # 初始化栅格地图
    occupancy_grid = np.zeros((nx, ny, nz), dtype=bool)

    # 点云映射到 Voxel Grid
    for (x, y, z) in points_world:
        ix = int((x - x_min) / resolution)
        iy = int((y - y_min) / resolution)
        iz = int((z - z_min) / resolution)

        # 设置占用标志
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            occupancy_grid[ix, iy, iz] = True

    return occupancy_grid, (x_min, y_min, z_min), (nx, ny, nz)


import numpy as np
import matplotlib.pyplot as plt


def visualize_voxel_grid(occupancy_grid, origin, resolution, edge_color=None):
    """
    可视化 Voxel Grid（体素网格）
    - 立方体填充
    - 颜色随高度变化（蓝色到黄色）
    - 透明度随高度增加
    - 可选去除黑色边线
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取占用体素的坐标
    occupied_indices = np.argwhere(occupancy_grid)
    if occupied_indices.shape[0] == 0:
        print("无可视化数据：体素网格为空")
        return

    X = occupied_indices[:, 0] * resolution + origin[0]
    Y = occupied_indices[:, 1] * resolution + origin[1]
    Z = occupied_indices[:, 2] * resolution + origin[2]

    # 计算高度范围，归一化颜色
    z_min, z_max = np.min(Z), np.max(Z)
    norm = plt.Normalize(z_min, z_max)
    colors = plt.cm.viridis(norm(Z))  # 颜色渐变

    # 透明度计算（高度越高透明度越大）
    alphas = 1 - (Z - z_min) / (z_max - z_min) * 0.3  # 最大透明度 0.7
    alphas = np.clip(alphas, 0.3, 1)  # 透明度范围 [0.3, 1]

    # 创建体素网格
    voxel_grid = np.zeros_like(occupancy_grid, dtype=bool)
    voxel_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]] = True

    # 生成正确的 RGBA 颜色网格
    facecolors = np.zeros((*occupancy_grid.shape, 4))  # 4通道 (RGBA)
    facecolors[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2], :3] = colors[:, :3]  # RGB
    facecolors[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2], 3] = alphas  # Alpha透明度

    # 选择边线颜色（默认无边线）
    edge_kw = {} if edge_color is None else {"edgecolors": edge_color, "linewidth": 0.1}

    # 绘制体素
    ax.voxels(voxel_grid, facecolors=facecolors, **edge_kw)

    # 轴标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Voxel Grid with Height-Based Color & Transparency")
    ax.view_init(elev=30, azim=-120)
    plt.show()


def main():
    """
    主程序: 获取点云 -> 构建体素地图 -> 可视化
    """
    # 1. 初始化无人机控制器
    drone = AirSimDroneController()

    # 2. 起飞
    drone.takeoff(flight_height=1.5)

    # 3. 采集点云数据
    print("采集点云中...")
    points_world = drone.create_point_cloud()

    if points_world is None or points_world.shape[0] == 0:
        print("点云采集失败，程序终止。")
        return

    # 4. 可视化原始点云
    visualize_3d_cloud(points_world)

    # 绘制点云
    R_world_to_show = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    points_world = (R_world_to_show @ points_world.T).T

    # 5. 构建体素地图
    print("构建体素地图...")
    occupancy_grid, origin, shape = create_voxel_grid(points_world, resolution=0.1)

    print(f"体素地图尺寸: {shape}, 原点: {origin}")

    # 6. 可视化体素地图
    # 去掉边线edge_color=None、灰色细边框edge_color='gray'、更细的边线edge_color='lightgray'
    visualize_voxel_grid(occupancy_grid, origin, resolution=0.1, edge_color='lightgray')

    # 7. 降落并释放控制
    drone.land_and_release()


if __name__ == "__main__":
    main()
