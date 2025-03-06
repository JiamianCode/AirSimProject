import heapq
import numpy as np


class AStarPathfinder:
    def __init__(self, resolution=0.1, safety_margin=0.2):
        """自定义 A* 搜索器"""
        self.resolution = resolution  # 体素网格分辨率
        self.safety_margin = safety_margin  # 安全边界
        self.occupancy_grid = {}  # 占用栅格（存储障碍物）
        self.origin = None  # 地图原点
        self.shape = None  # 栅格地图大小

    def set_occupancy_grid(self, voxel_grid):
        """从 VoxelGridManager 生成占用地图"""
        self.occupancy_grid = {key: True for key in voxel_grid.global_voxel_grid.grid.keys()}
        print(f"占用栅格地图已生成，共 {len(self.occupancy_grid)} 个障碍物体素")

    def world_to_grid(self, x, y, z):
        """世界坐标 -> 栅格坐标"""
        ix = int(np.floor(x / self.resolution))
        iy = int(np.floor(y / self.resolution))
        iz = int(np.floor(z / self.resolution))
        return ix, iy, iz

    def grid_to_world(self, ix, iy, iz):
        """栅格坐标 -> 世界坐标"""
        x = ix * self.resolution
        y = iy * self.resolution
        z = iz * self.resolution
        return x, y, z

    def is_valid(self, ix, iy, iz):
        """检查该栅格是否有效（未被占用）"""
        return (ix, iy, iz) not in self.occupancy_grid

    def search(self, start, goal):
        """ 运行 A* 搜索算法  """
        start_grid = self.world_to_grid(*start)
        goal_grid = self.world_to_grid(*goal)

        if not self.is_valid(*start_grid):
            print("起点位于障碍物内，无法规划路径")
            return None

        if not self.is_valid(*goal_grid):
            print("终点位于障碍物内，无法规划路径")
            return None

        print(f"A* 规划: 从 {start_grid} 到 {goal_grid}")

        # A* 初始化
        open_set = [(0, start_grid)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: np.linalg.norm(np.array(start_grid) - np.array(goal_grid))}

        neighbors = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)  # 六方向移动
        ]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                # 路径回溯
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(start)  # 加入起点
                path.reverse()
                print(f"路径找到，共 {len(path)} 个点")
                return path

            for dx, dy, dz in neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not self.is_valid(*neighbor):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal_grid))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

        print("无可行路径")
        return None
