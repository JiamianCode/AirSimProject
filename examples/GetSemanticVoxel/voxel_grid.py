import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


class Voxel:
    """表示一个体素，只记录该体素内的点云数量"""
    def __init__(self):
        self.point_count = 0  # 记录每个网格内的点数

    def increment_point_count(self):
        """增加点云数量"""
        self.point_count += 1


class InstanceGrid:
    """表示不同实例对应的网格，包括实例网格的体素和点云"""
    def __init__(self, label=None, object_id=None):
        self.label = label  # 实例标签
        self.object_id = object_id  # 实例的原始 object_id
        self.grid = set()  # 存储体素网格索引，使用 set 存储体素索引
        self.points = []  # 存储体素的点云坐标

    def add_voxel(self, voxel_key, point):
        """将体素添加到实例网格，并记录点云坐标"""
        # 将体素的索引添加到 set 中
        self.grid.add(voxel_key)
        # 存储点云坐标
        self.points.append(point)


class VoxelGrid:
    """表示局部体素网格"""
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.grid = {}  # 定义网格，使用 dict 存储体素

    def get_voxel_key(self, position):
        """根据体素的空间位置返回体素的键值，体素坐标系与原点对齐"""
        voxel_key = tuple(np.floor(position / self.voxel_size).astype(int))
        return voxel_key

    def add_point(self, point):
        """将点添加到体素网格中"""
        voxel_key = self.get_voxel_key(point)
        if voxel_key not in self.grid:
            self.grid[voxel_key] = Voxel()
        self.grid[voxel_key].increment_point_count()  # 增加该体素的点数

    def get_voxel(self, position):
        """获取指定位置的体素"""
        voxel_key = self.get_voxel_key(position)
        return self.grid.get(voxel_key, None)

    def create_voxel_map(self, points, valid_indices, candidate_labels, semantic_img, label_manager):
        """根据点云生成体素地图并赋予语义标签"""
        instance_grids = {}

        # 遍历每个点，处理点云和语义
        for idx in range(valid_indices.shape[0]):
            u, v = valid_indices[idx]
            point = points[idx]
            object_id = list(semantic_img[v, u])

            # 如果点云属于候选标签，则将该点添加到相应的实例网格
            label = None
            for candidate_label in candidate_labels:
                if object_id in label_manager.label_to_ids.get(candidate_label, []):
                    label = candidate_label
                    break

            # 如果在候选标签中
            if label:
                # 添加到体素网格，并且在实例网格中注册体素索引
                self.add_point(point)
                if label not in instance_grids:
                    instance_grids[label] = InstanceGrid(label=label, object_id=object_id)

                voxel_key = self.get_voxel_key(point)
                instance_grids[label].add_voxel(voxel_key, point)
            else:
                # 如果不在候选标签中，标记为障碍物
                self.add_point(point)

        return instance_grids


class VoxelGridManager:
    """管理全局体素网格和实例网格，提供合并和可视化功能"""
    def __init__(self, voxel_size=0.1):
        self.global_voxel_grid = VoxelGrid(voxel_size=voxel_size)
        self.global_origin = np.array([0.0, 0.0, 0.0])  # 坐标系原点固定为 (0, 0, 0)
        self.global_instance_grids = {}

    def create_and_merge_local_map(self, points, valid_indices, candidate_labels, semantic_img, label_manager):
        """创建局部体素网格并合并到全局体素网格"""
        # 创建局部体素网格
        local_voxel_grid = VoxelGrid(voxel_size=self.global_voxel_grid.voxel_size)
        local_instance_grids = local_voxel_grid.create_voxel_map(
            points, valid_indices, candidate_labels, semantic_img, label_manager
        )

        # 统计障碍物网格和实例网格的增加数量
        obstacle_count = 0
        instance_count = 0

        # 合并局部体素网格到全局体素网格
        for voxel_key, voxel in local_voxel_grid.grid.items():
            if voxel_key not in self.global_voxel_grid.grid:
                self.global_voxel_grid.grid[voxel_key] = voxel
                obstacle_count += 1  # 增加的体素是障碍物
            else:
                self.global_voxel_grid.grid[voxel_key].point_count += voxel.point_count

        # 合并实例网格
        for label, local_instance_grid in local_instance_grids.items():
            if label not in self.global_instance_grids:
                self.global_instance_grids[label] = local_instance_grid
                instance_count += 1  # 新实例网格增加
            else:
                self.global_instance_grids[label].grid.update(local_instance_grid.grid)
                self.global_instance_grids[label].points.extend(local_instance_grid.points)

        # 计算网格变化量
        grid_change = obstacle_count + instance_count
        return grid_change

    def visualize(self, remove_top=False):
        """可视化全局体素网格，支持去顶操作"""
        occupied_indices = np.array(list(self.global_voxel_grid.grid.keys()))
        if occupied_indices.shape[0] == 0:
            print("无可视化数据：体素网格为空")
            return

        # 为每个体素初始化颜色，默认是灰色
        colors = np.full((occupied_indices.shape[0], 3), [128, 128, 128], dtype=np.uint8)  # 默认所有体素为灰色

        # 如果 remove_top 为 True，进行去顶操作
        if remove_top:
            # 获取所有体素点的 z 值
            z_values = occupied_indices[:, 2] * self.global_voxel_grid.voxel_size + self.global_origin[2]

            # 计算 z_min 和 z_max
            z_min = np.min(z_values)
            z_max = np.max(z_values)

            # 计算去顶的高度范围 (从 z_max 开始，保留 75% 的高度)
            z_min_cutoff = z_max - 0.75 * (z_max - z_min)  # 保留 80% 的高度，去掉顶部 20%

            # 只保留在此范围内的点
            valid_indices = z_values >= z_min_cutoff
            occupied_indices = occupied_indices[valid_indices]
            colors = colors[valid_indices]  # 保证颜色数组与筛选后的点云一致

        # 遍历实例，首先可视化实例的体素
        for instance_grid in self.global_instance_grids.values():
            instance_color = instance_grid.object_id  # 获取实例的颜色

            # 遍历实例网格中的体素，并修改颜色
            for voxel_key in instance_grid.grid:
                # 更新所有匹配的体素颜色
                idx = np.where(np.all(occupied_indices == voxel_key, axis=1))[0]  # 获取所有匹配的体素索引
                colors[idx] = instance_color  # 修改为实例的颜色

        # 提取 X, Y, Z 坐标
        X = occupied_indices[:, 0] * self.global_voxel_grid.voxel_size + self.global_origin[0]
        Y = occupied_indices[:, 1] * self.global_voxel_grid.voxel_size + self.global_origin[1]
        Z = occupied_indices[:, 2] * self.global_voxel_grid.voxel_size + self.global_origin[2]

        # 可视化体素网格
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c=colors / 255, marker='o', alpha=0.5)

        # 设置轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Voxel Grid with Instances")
        ax.view_init(elev=190, azim=-20)  # 观察视角，z轴设置为向下

        plt.show()
