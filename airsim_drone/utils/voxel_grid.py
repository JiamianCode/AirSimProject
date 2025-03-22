import numpy as np
import torch

from airsim_drone.visualization.visualize_global_voxel_grid import visualize_global_voxel_grid


class Voxel:
    """表示一个体素，仅记录该体素内的点云数量"""
    def __init__(self):
        self.point_count = 0  # 记录体素内的点数

    def increment_point_count(self, count=1):
        """增加体素内的点云数量"""
        self.point_count += count


class VoxelGrid:
    """表示局部体素网格"""
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.grid = {}  # 使用字典存储体素

    def get_voxel_keys_batch(self, points):
        """使用 PyTorch 计算所有点的体素索引，避免 for 循环"""
        points_tensor = torch.tensor(points, dtype=torch.float32)  # 转换为 Tensor
        voxel_keys = torch.floor(points_tensor / self.voxel_size).to(torch.int).numpy()
        return [tuple(key) for key in voxel_keys]  # 转换为 Python tuple 作为哈希表索引

    def get_voxel_center(self, voxel_key):
        """返回给定体素的中心坐标"""
        x, y, z = voxel_key
        center_x = (x + 0.5) * self.voxel_size
        center_y = (y + 0.5) * self.voxel_size
        center_z = (z + 0.5) * self.voxel_size
        return np.array([center_x, center_y, center_z])

    def add_points(self, points):
        """批量将点云数据添加到体素网格"""
        voxel_keys = self.get_voxel_keys_batch(points)

        # 统计每个体素的点数
        voxel_counts = {}
        for key in voxel_keys:
            if key in voxel_counts:
                voxel_counts[key] += 1
            else:
                voxel_counts[key] = 1

        # 更新体素网格
        for key, count in voxel_counts.items():
            if key not in self.grid:
                self.grid[key] = Voxel()
            self.grid[key].increment_point_count(count)

    def create_voxel_map(self, points):
        """批量转换点云为体素网格"""
        self.add_points(points)


class VoxelGridManager:
    """管理全局体素网格，提供合并和可视化功能"""
    def __init__(self, voxel_size=0.1):
        self.global_voxel_grid = VoxelGrid(voxel_size=voxel_size)
        self.global_origin = np.array([0.0, 0.0, 0.0])  # 坐标系原点

    def create_and_merge_local_map(self, points, visualize=False):
        """创建局部体素网格并合并到全局体素网格"""
        local_voxel_grid = VoxelGrid(voxel_size=self.global_voxel_grid.voxel_size)
        local_voxel_grid.create_voxel_map(points)

        # 合并局部体素网格到全局体素网格
        for voxel_key, voxel in local_voxel_grid.grid.items():
            if voxel_key not in self.global_voxel_grid.grid:
                self.global_voxel_grid.grid[voxel_key] = voxel
            else:
                self.global_voxel_grid.grid[voxel_key].point_count += voxel.point_count

        if visualize:
            visualize_global_voxel_grid(self)
