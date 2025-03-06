import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


class Voxel:
    """表示一个体素，仅记录该体素内的点云数量"""

    def __init__(self):
        self.point_count = 0  # 记录体素内的点数

    def increment_point_count(self):
        """增加体素内的点云数量"""
        self.point_count += 1


class VoxelGrid:
    """表示局部体素网格"""

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.grid = {}  # 使用字典存储体素

    def get_voxel_key(self, position):
        """根据点的位置计算其体素的索引"""
        voxel_key = tuple(np.floor(position / self.voxel_size).astype(int))
        return voxel_key

    def add_point(self, point):
        """将点添加到体素网格中"""
        voxel_key = self.get_voxel_key(point)
        if voxel_key not in self.grid:
            self.grid[voxel_key] = Voxel()
        self.grid[voxel_key].increment_point_count()  # 增加该体素的点数

    def create_voxel_map(self, points):
        """根据点云数据创建体素网格"""
        for point in points:
            self.add_point(point)


class VoxelGridManager:
    """管理全局体素网格，提供合并和可视化功能"""

    def __init__(self, voxel_size=0.1):
        self.global_voxel_grid = VoxelGrid(voxel_size=voxel_size)
        self.global_origin = np.array([0.0, 0.0, 0.0])  # 坐标系原点

    def create_and_merge_local_map(self, points):
        """创建局部体素网格并合并到全局体素网格"""
        # 创建局部体素网格
        local_voxel_grid = VoxelGrid(voxel_size=self.global_voxel_grid.voxel_size)
        local_voxel_grid.create_voxel_map(points)

        # 合并局部体素网格到全局体素网格
        for voxel_key, voxel in local_voxel_grid.grid.items():
            if voxel_key not in self.global_voxel_grid.grid:
                self.global_voxel_grid.grid[voxel_key] = voxel
            else:
                self.global_voxel_grid.grid[voxel_key].point_count += voxel.point_count

    def visualize(self, remove_top=True):
        """可视化全局体素网格，支持去顶操作"""
        occupied_indices = np.array(list(self.global_voxel_grid.grid.keys()))
        if occupied_indices.shape[0] == 0:
            print("无可视化数据：体素网格为空")
            return

        # 默认体素颜色为灰色
        colors = np.full((occupied_indices.shape[0], 3), [128, 128, 128], dtype=np.uint8)

        # 如果 remove_top 为 True，进行去顶操作
        if remove_top:
            z_values = occupied_indices[:, 2] * self.global_voxel_grid.voxel_size + self.global_origin[2]
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            z_min_cutoff = z_max - 0.75 * (z_max - z_min)  # 保留 75% 的高度

            # 只保留在此范围内的点
            valid_indices = z_values >= z_min_cutoff
            occupied_indices = occupied_indices[valid_indices]
            colors = colors[valid_indices]  # 颜色数组同步调整

        # 提取 X, Y, Z 坐标
        X = occupied_indices[:, 0] * self.global_voxel_grid.voxel_size + self.global_origin[0]
        Y = occupied_indices[:, 1] * self.global_voxel_grid.voxel_size + self.global_origin[1]
        Z = occupied_indices[:, 2] * self.global_voxel_grid.voxel_size + self.global_origin[2]

        # 可视化体素网格
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8)

        # 设置轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Voxel Grid")
        ax.view_init(elev=190, azim=-20)  # 设置视角

        plt.show()
