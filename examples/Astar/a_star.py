import heapq
import math
import numpy as np


class AStar3D:
    def __init__(self, resolution=0.1, safety_margin=0.1):
        """
        初始化 A* 3D 搜索器
        """
        self.resolution = resolution
        self.safety_margin = safety_margin
        self.occupancy_grid = None
        self.origin = None
        self.shape = None

    @staticmethod
    def world_to_grid(x, y, z, origin, resolution):
        """世界坐标 -> 栅格坐标"""
        x_min, y_min, z_min = origin
        ix = int((x - x_min) // resolution)
        iy = int((y - y_min) // resolution)
        iz = int((z - z_min) // resolution)
        return ix, iy, iz

    @staticmethod
    def grid_to_world(ix, iy, iz, origin, resolution):
        """栅格坐标 -> 世界坐标"""
        x_min, y_min, z_min = origin
        x = ix * resolution + x_min + resolution / 2.0
        y = iy * resolution + y_min + resolution / 2.0
        z = iz * resolution + z_min + resolution / 2.0
        return x, y, z

    def is_valid(self, ix, iy, iz):
        """检查栅格是否在地图范围内且未被占用"""
        nx, ny, nz = self.shape
        return 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and not self.occupancy_grid[ix, iy, iz]

    def find_nearest_valid_grid(self, start_world, goal_world):
        """在起点到终点的连线上寻找最近的可用栅格"""
        start_grid = np.array(self.world_to_grid(*start_world, self.origin, self.resolution))
        goal_grid = np.array(self.world_to_grid(*goal_world, self.origin, self.resolution))

        direction = goal_grid - start_grid
        length = np.linalg.norm(direction)
        if length == 0:
            return None  # 起点和终点相同

        step = direction / length
        num_steps = int(length)

        for i in range(num_steps + 1):
            current_grid = np.round(start_grid + i * step).astype(int)
            ix, iy, iz = current_grid

            # 如果当前点在栅格内，且未被占用，则返回
            if self.is_valid(ix, iy, iz):
                return tuple(current_grid)

            # 如果被占用，尝试向上搜索 0.2m 直到找到可用点
            max_up_steps = int(0.5 / self.resolution)  # 最多向上移动0.5m
            for up in range(1, max_up_steps + 1):
                iz_up = iz + up
                if self.is_valid(ix, iy, iz_up):
                    return ix, iy, iz_up

        return None  # 无法找到可用栅格

    def create_occupancy_grid(self, points_world):
        """
        根据点云构建 3D 占用栅格地图。
        """
        if len(points_world) == 0:
            raise ValueError("点云为空，无法构建栅格地图")

        # 获取点云边界
        x_min, x_max = np.min(points_world[:, 0]), np.max(points_world[:, 0])
        y_min, y_max = np.min(points_world[:, 1]), np.max(points_world[:, 1])
        z_min, z_max = np.min(points_world[:, 2]), np.max(points_world[:, 2])

        # 增加额外边界以防止路径受限
        margin = 0.3
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        z_min -= margin
        z_max += margin

        # 计算栅格大小
        nx = int(math.ceil((x_max - x_min) / self.resolution))
        ny = int(math.ceil((y_max - y_min) / self.resolution))
        nz = int(math.ceil((z_max - z_min) / self.resolution))

        print(f"3D 栅格范围 X:[{x_min:.2f},{x_max:.2f}], Y:[{y_min:.2f},{y_max:.2f}], Z:[{z_min:.2f},{z_max:.2f}]")
        print(f"3D 栅格大小: nx={nx}, ny={ny}, nz={nz} (每格{self.resolution}m)")

        # 初始化栅格地图
        occupancy_grid = np.zeros((nx, ny, nz), dtype=bool)

        # 计算障碍物扩展范围
        margin_cells = int(math.ceil(self.safety_margin / self.resolution))

        # 设定占用栅格
        for (x, y, z) in points_world:
            ix, iy, iz = self.world_to_grid(x, y, z, (x_min, y_min, z_min), self.resolution)
            for dx in range(-margin_cells, margin_cells + 1):
                for dy in range(-margin_cells, margin_cells + 1):
                    for dz in range(-margin_cells, margin_cells + 1):
                        nx_, ny_, nz_ = ix + dx, iy + dy, iz + dz
                        if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                            occupancy_grid[nx_, ny_, nz_] = True

        self.occupancy_grid = occupancy_grid
        self.origin = (x_min, y_min, z_min)
        self.shape = (nx, ny, nz)

    def search(self, start_world, goal_world):
        """运行 A* 3D 搜索算法"""
        if self.occupancy_grid is None:
            raise RuntimeError("请先构建 3D 栅格地图")

        # 终点坐标适当抬升，以免被作为障碍而无法到达
        goal_world[2] -= 0.15  # z轴向下为正

        # 处理起点不在栅格内的情况
        start_cell = self.world_to_grid(*start_world, self.origin, self.resolution)
        goal_cell = self.world_to_grid(*goal_world, self.origin, self.resolution)

        if not self.is_valid(*start_cell):
            print("起点超出栅格范围，寻找最近可用栅格...")
            new_start_cell = self.find_nearest_valid_grid(start_world, goal_world)
            if new_start_cell is None:
                print("找不到合适的起点，无法规划路径")
                return None
            new_start_world = self.grid_to_world(*new_start_cell, self.origin, self.resolution)
            print(f"选择新的起点 {new_start_cell} 对应世界坐标 {new_start_world}")
        else:
            new_start_cell = start_cell
            new_start_world = start_world

        # 终点检查
        if not self.is_valid(*goal_cell):
            print("终点无效：被障碍物占据或超出范围")
            return None

        print(f"规划路径: 起点 {new_start_cell}, 终点 {goal_cell}")

        # A* 初始化
        neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        open_set = [(0, new_start_cell)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {new_start_cell: 0.0}
        f_score = {new_start_cell: math.dist(new_start_cell, goal_cell)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_cell:
                # 回溯路径
                path_cells = []
                while current in came_from:
                    path_cells.append(current)
                    current = came_from[current]
                path_cells.append(new_start_cell)
                path_cells.reverse()

                # 转换路径到世界坐标
                path_world = [self.grid_to_world(ix, iy, iz, self.origin, self.resolution) for (ix, iy, iz) in
                              path_cells]

                # 连接原始起点到新起点（如果新起点不同于原始起点）
                if not np.allclose(new_start_world, start_world):
                    path_world = [start_world] + path_world

                print(f"路径找到，长度: {len(path_world)}")
                return path_world

            for dx, dy, dz in neighbors_6:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not self.is_valid(*neighbor):
                    continue

                tentative_g_score = g_score[current] + 1.0
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + math.dist(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

        print("无可行路径")
        return None
