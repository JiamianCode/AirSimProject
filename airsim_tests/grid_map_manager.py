import datetime
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import airsim

matplotlib.use('TkAgg')


class GridMapManager:
    """
    2D 栅格地图管理器，存储和可视化：
    - 墙体
    - 可行区域（无人机视野范围）
    - 视线扫射（基于相机视野）
    - 视野边界内的空缺填补
    - 栅格更新次数控制
    - 近距离墙体保护（min_range）
    """

    def __init__(self, grid_size=0.1):
        """
        初始化栅格地图。
        :param grid_size: 每个栅格的大小 (单位: 米)
        """
        self.grid_size = grid_size  # 栅格大小
        self.grid_map = {}  # 存储墙体 & 可行区域 (key: (i, j), value: 1=墙, 0=可行)
        self.grid_count = {}  # 存储栅格更新次数
        self.grid_normals = {}  # 存储墙体法向量 (key: (i, j), value: np.array([nx, ny, nz]))
        self.scanned_area = set()  # 记录已扫描的区域

    def world_to_grid(self, x, y):
        """将世界坐标 (x, y) 转换为栅格坐标。"""
        i = int(x / self.grid_size)
        j = int(y / self.grid_size)
        return i, j

    def extract_wall_map(self, extracted_planes, drone_pos, max_range):
        """
        过滤点云数据并栅格化标记墙体，不计算法向量。

        :param extracted_planes: 提取的墙体点云数据
        :param drone_pos: 无人机的当前位置 (x, y)
        :param max_range: 处理点云的最大距离
        :return: dict - 仅包含墙体的 2D 栅格地图 (key: 栅格坐标, value: 1)
        """
        new_map = {}

        drone_x, drone_y = drone_pos

        for plane in extracted_planes:
            if "inlier_points" in plane:
                points = plane["inlier_points"]
                if isinstance(points, torch.Tensor):
                    points = points.cpu().numpy()  # 转换为 NumPy 数组

                # 筛选距离在 max_range 以内的点
                distances = np.linalg.norm(points[:, :2] - np.array([drone_x, drone_y]), axis=1)
                valid_points = points[distances <= max_range]

                # 进行栅格化，将墙体点投影到 2D 栅格地图
                for point in valid_points:
                    x, y, _ = point
                    grid_pos = self.world_to_grid(x, y)
                    new_map[grid_pos] = 1  # 标记墙体

        return new_map

    def create_and_merge(self, extracted_planes, drone_pos, drone_orientation, fov,
                         max_range=8.0, min_range=2.0, gap_threshold=5, visualize=False):
        """
        处理墙体点云数据并合并至栅格地图
        """
        # **提取墙体信息，进行栅格化**
        new_map = self.extract_wall_map(extracted_planes, drone_pos, max_range)

        # **获取无人机偏航角（Yaw）**
        roll, pitch, yaw = airsim.to_eularian_angles(drone_orientation)

        # **视野范围扫射**
        self.ray_scan(drone_pos, yaw, fov, max_range, new_map)

        # **填充优化**
        self.fill_gaps(new_map)

        # **计算法向量**
        self.compute_wall_normals(new_map)

        self.grid_map = new_map

        # **可视化（可选）**
        if visualize:
            self.visualize_map(drone_pos, yaw)
        else:
            self.save(drone_pos)

    def compute_wall_normals(self, new_map, gap_threshold=10):
        """
        计算可行区域的法向量，然后基于这些法向量重新填充墙栅格，使用插值填补墙体缺口，最后填充探索前沿。

        :param new_map: 2D 栅格地图，包含可行区域 (0) 和墙体 (1)
        :param gap_threshold: 插值填补墙体缺口的最大距离
        """
        # 清空旧的法向量数据
        self.grid_normals = {}

        # 记录可行区域的法向量
        free_space_normals = {}

        # **第一步：计算可行区域的梯度作为法向量**
        for (grid_x, grid_y) in new_map.keys():
            if new_map[(grid_x, grid_y)] == 0:  # 仅计算可行区域
                neighbors = [
                    (grid_x - 1, grid_y), (grid_x + 1, grid_y),  # 左右
                    (grid_x, grid_y - 1), (grid_x, grid_y + 1),  # 上下
                    (grid_x - 1, grid_y - 1), (grid_x - 1, grid_y + 1),  # 左上、左下
                    (grid_x + 1, grid_y - 1), (grid_x + 1, grid_y + 1)  # 右上、右下
                ]

                gradient_x, gradient_y = 0, 0

                for (nx, ny) in neighbors:
                    if (nx, ny) not in new_map or new_map[(nx, ny)] == 1:  # 遇到墙或未知区域
                        gradient_x += nx - grid_x
                        gradient_y += ny - grid_y

                # 计算法向量（取负梯度方向）
                normal = np.array([-gradient_x, -gradient_y])
                norm = np.linalg.norm(normal)
                if norm > 0:
                    free_space_normals[(grid_x, grid_y)] = normal / norm  # 归一化

        # **第二步：仅在邻近已有墙体的地方填充墙体**
        new_wall_map = {}  # 新的墙体栅格

        for (grid_x, grid_y), normal in free_space_normals.items():
            neighbors = [
                (grid_x - 1, grid_y), (grid_x + 1, grid_y),
                (grid_x, grid_y - 1), (grid_x, grid_y + 1),
                (grid_x - 1, grid_y - 1), (grid_x - 1, grid_y + 1),
                (grid_x + 1, grid_y - 1), (grid_x + 1, grid_y + 1)
            ]

            for (nx, ny) in neighbors:
                if (nx, ny) not in new_map or new_map[(nx, ny)] == 1:  # 原本是墙或未知区域
                    # **确保只有邻近已有墙体的区域才填充墙体**
                    has_adjacent_wall = any(
                        (nnx, nny) in new_map and new_map[(nnx, nny)] == 1
                        for nnx, nny in [
                            (nx - 1, ny), (nx + 1, ny),
                            (nx, ny - 1), (nx, ny + 1)
                        ]
                    )

                    if has_adjacent_wall:  # 只有当该区域邻近已有墙体时，才填充墙
                        new_wall_map[(nx, ny)] = 1  # 重新标记墙体
                        self.grid_normals[(nx, ny)] = normal  # 赋予墙体法向量

        # **用新的墙体栅格替换旧的墙**
        for key in list(new_map.keys()):
            if new_map[key] == 1:
                del new_map[key]  # 移除旧的墙栅格

        new_map.update(new_wall_map)  # 更新墙体

        # **第三步：填补墙体缺口**
        self.interpolate_walls(gap_threshold)

        # **第四步：填充探索前沿**
        new_frontier_map = {}  # 新的探索前沿栅格

        for (grid_x, grid_y), normal in free_space_normals.items():
            neighbors = [
                (grid_x - 1, grid_y), (grid_x + 1, grid_y),
                (grid_x, grid_y - 1), (grid_x, grid_y + 1),
                (grid_x - 1, grid_y - 1), (grid_x - 1, grid_y + 1),
                (grid_x + 1, grid_y - 1), (grid_x + 1, grid_y + 1)
            ]

            for (nx, ny) in neighbors:
                if (nx, ny) not in new_map or new_map[(nx, ny)] == 1:  # 原本是墙或未知区域
                    has_adjacent_wall = any(
                        (nnx, nny) in new_map and new_map[(nnx, nny)] == 1
                        for nnx, nny in [
                            (nx - 1, ny), (nx + 1, ny),
                            (nx, ny - 1), (nx, ny + 1)
                        ]
                    )

                    if not has_adjacent_wall:  # 只在墙体不相邻的地方填充探索前沿
                        new_frontier_map[(nx, ny)] = 2  # 2 表示探索前沿
                        self.grid_normals[(nx, ny)] = -normal  # 赋予探索前沿法向量（方向相反朝外）

        new_map.update(new_frontier_map)  # 更新探索前沿

    def ray_scan(self, drone_pos, drone_yaw, fov, max_range, map_to_update):
        """视野内扫射，检测可行区域"""
        min_angle = drone_yaw - np.radians(fov / 2)
        max_angle = drone_yaw + np.radians(fov / 2)
        angles = np.linspace(min_angle, max_angle, 50)

        for theta in angles:
            x_step = np.cos(theta) * self.grid_size
            y_step = np.sin(theta) * self.grid_size
            x, y = drone_pos

            for _ in range(int(max_range / self.grid_size)):
                grid_pos = self.world_to_grid(x, y)

                if grid_pos in map_to_update and map_to_update[grid_pos] == 1:
                    break  # 遇到墙体，停止扫描

                if grid_pos not in map_to_update and self.has_adjacent_wall(grid_pos, map_to_update):
                    break  # 遇到墙体邻近区域，停止扫描

                map_to_update[grid_pos] = 0  # 标记可行区域
                x += x_step
                y += y_step

    def has_adjacent_wall(self, grid_pos, map_to_check):
        """检查某个栅格的 8 邻域是否有墙"""
        i, j = grid_pos
        neighbors = [
            (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
            (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
        ]

        return any(n in map_to_check and map_to_check[n] == 1 for n in neighbors)

    def fill_gaps(self, map_to_update):
        """填充空缺区域，包括可行区域，并确保可行区域可以紧贴墙体"""
        new_fill = []

        for grid_pos in list(map_to_update.keys()):
            if map_to_update[grid_pos] != 0:  # 只处理可行区域
                continue

            i, j = grid_pos
            neighbors = [
                (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)
            ]

            for n in neighbors:
                if n not in map_to_update:  # 发现未填充区域
                    new_fill.append(n)
                elif map_to_update[n] == 1:  # 发现墙体，只要当前格子不是墙体，就填充
                    new_fill.append(grid_pos)

        # 批量填充新可行区域
        for pos in new_fill:
            if pos not in map_to_update:  # 避免覆盖墙体
                map_to_update[pos] = 0

    def interpolate_walls(self, gap_threshold):
        """填补墙体缺口，并插值法向信息"""
        wall_positions = [pos for pos, value in self.grid_map.items() if value == 1]

        for i, pos1 in enumerate(wall_positions):
            for j, pos2 in enumerate(wall_positions):
                if i >= j:  # 避免重复计算
                    continue

                # **计算两点之间的曼哈顿距离**
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

                if 1 < dist <= gap_threshold:  # 仅填补小缺口
                    # **插值填补墙体**
                    interp_positions = self.bresenham_line(pos1[0], pos1[1], pos2[0], pos2[1])
                    normal1 = self.grid_normals.get(pos1, np.array([1, 0]))
                    normal2 = self.grid_normals.get(pos2, np.array([1, 0]))

                    for idx, interp_pos in enumerate(interp_positions):
                        if interp_pos not in self.grid_map:
                            self.grid_map[interp_pos] = 1
                            alpha = idx / len(interp_positions)
                            interp_normal = (1 - alpha) * normal1 + alpha * normal2
                            self.grid_normals[interp_pos] = interp_normal / np.linalg.norm(interp_normal)

    def bresenham_line(self, x1, y1, x2, y2):
        """使用 Bresenham 直线算法计算从 (x1, y1) 到 (x2, y2) 之间的所有网格点"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points

    def visualize_map(self, drone_pos, drone_yaw):
        """可视化栅格地图，并绘制墙体法向量、无人机朝向和探索前沿"""
        if not self.grid_map:
            print("地图为空，无法可视化")
            return

        wall_coords = np.array([k for k, v in self.grid_map.items() if v == 1])
        free_coords = np.array([k for k, v in self.grid_map.items() if v == 0])
        frontier_coords = np.array([k for k, v in self.grid_map.items() if v == 2])  # 取探索前沿
        drone_grid = self.world_to_grid(*drone_pos)

        plt.figure(figsize=(10, 10))
        plt.xlabel("Y (Grid)")
        plt.ylabel("X (Grid)")
        plt.title("2D Grid Map with Wall Normals, Drone Orientation, and Frontier")
        plt.grid(True, linestyle="--", alpha=0.5)

        # 画墙体
        if wall_coords.size > 0:
            plt.scatter(wall_coords[:, 1], wall_coords[:, 0], c="black", marker="s", label="Walls", s=50)

        # 画可行区域
        if free_coords.size > 0:
            plt.scatter(free_coords[:, 1], free_coords[:, 0], c="lightgray", marker="s", label="Free Space", s=30)

        # 画探索前沿
        if frontier_coords.size > 0:
            plt.scatter(frontier_coords[:, 1], frontier_coords[:, 0], c="yellow", marker="s", label="Frontier", s=50)

        # 画无人机位置
        plt.scatter(drone_grid[1], drone_grid[0], c="red", marker="o", s=100, label="Drone")

        # 画墙体法向量 & 探索前沿法向量
        for (grid_x, grid_y), normal in self.grid_normals.items():
            start_x, start_y = grid_y, grid_x  # 交换坐标以匹配绘图
            end_x = start_x + normal[1] * 0.5  # 法向量箭头长度
            end_y = start_y + normal[0] * 0.5

            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

        # 画无人机朝向箭头
        arrow_length = 1.0  # 设置无人机朝向箭头的长度
        yaw_x = np.cos(drone_yaw) * arrow_length
        yaw_y = np.sin(drone_yaw) * arrow_length

        plt.arrow(drone_grid[1], drone_grid[0], yaw_y, yaw_x, head_width=0.2, head_length=0.2, fc='green', ec='green', label="Drone Orientation")

        plt.legend()
        plt.show()

    def save(self, drone_pos):
        """可视化栅格地图并保存为图片"""
        if not self.grid_map:
            print("地图为空，无法可视化")
            return

        wall_coords = np.array([k for k, v in self.grid_map.items() if v == 1])
        free_coords = np.array([k for k, v in self.grid_map.items() if v == 0])
        drone_grid = self.world_to_grid(*drone_pos)

        plt.figure(figsize=(10, 10))
        plt.xlabel("Y (Grid)")
        plt.ylabel("X (Grid)")
        plt.title("2D Grid Map (Walls, Free Space & Drone)")
        plt.grid(True, linestyle="--", alpha=0.5)

        if wall_coords.size > 0:
            plt.scatter(wall_coords[:, 1], wall_coords[:, 0], c="black", marker="s", label="Walls", s=50)

        if free_coords.size > 0:
            plt.scatter(free_coords[:, 1], free_coords[:, 0], c="lightgray", marker="s", label="Free Space", s=30)

        plt.scatter(drone_grid[1], drone_grid[0], c="red", marker="o", s=100, label="Drone")

        plt.legend()

        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"../output/fly_path/grid_map_{timestamp}.png"
        # 保存图像
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.abspath(os.path.join(script_dir, save_path))

        # 保存图片
        plt.savefig(absolute_path, dpi=300)
        plt.close()
