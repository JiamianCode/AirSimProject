import heapq
import numpy as np

class AStarPathfinder:
    def __init__(self, resolution=0.1, safety_margin=0.2):
        """自定义 A* 搜索器"""
        self.resolution = resolution  # 体素网格分辨率
        self.safety_margin = safety_margin  # 安全边界
        self.occupancy_grid = {}  # 存储障碍物

    def set_occupancy_grid(self, voxel_grid):
        """从 VoxelGridManager 生成占用地图"""
        self.occupancy_grid = {key: True for key in voxel_grid.global_voxel_grid.grid.keys()}
        print(f"占用栅格地图已生成，共 {len(self.occupancy_grid)} 个障碍物")

    def world_to_grid(self, x, y, z):
        """世界坐标 -> 栅格坐标"""
        return tuple(map(lambda v: int(np.floor(v / self.resolution)), (x, y, z)))

    def grid_to_world(self, ix, iy, iz):
        """栅格坐标 -> 世界坐标"""
        return ix * self.resolution, iy * self.resolution, iz * self.resolution

    def is_valid(self, ix, iy, iz):
        """检查该栅格是否有效（未被占用）"""
        return (ix, iy, iz) not in self.occupancy_grid

    def is_line_clear(self, start, goal):
        """检查从 start 到 goal 的直线路径是否可行"""
        start_grid = self.world_to_grid(*start)
        goal_grid = self.world_to_grid(*goal)
        steps = max(abs(goal_grid[i] - start_grid[i]) for i in range(3))  # 取最大步数
        if steps == 0:
            return True  # 已经在目标点

        for i in range(steps):
            t = i / steps
            mid_x = start[0] * (1 - t) + goal[0] * t
            mid_y = start[1] * (1 - t) + goal[1] * t
            mid_z = start[2] * (1 - t) + goal[2] * t
            if not self.is_valid(*self.world_to_grid(mid_x, mid_y, mid_z)):
                return False  # 有障碍
        return True

    def search(self, voxel_manager, start, goal):
        """ 运行优化版 A* 搜索算法  """
        self.set_occupancy_grid(voxel_manager)

        # **如果直线路径可行，直接返回直线路径**
        if self.is_line_clear(start, goal):
            return [start, goal]

        # **如果直线不可行，进行分段规划**
        start_grid = self.world_to_grid(*start)
        goal_grid = self.world_to_grid(*goal)

        if not self.is_valid(*start_grid):
            print("起点位于障碍物内，无法规划路径")
            return None

        if not self.is_valid(*goal_grid):
            print("终点位于障碍物内，无法规划路径")
            return None

        print(f"A* 规划: 从 {start_grid} 到 {goal_grid}")

        # **A* 初始化**
        open_set = [(0, start_grid)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: np.linalg.norm(np.array(start_grid) - np.array(goal_grid))}

        neighbors = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)  # 六方向移动
        ]

        # **A* 搜索**
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                # **路径回溯**
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(start)  # 加入起点
                path.reverse()

                # **尝试连接直线段**
                optimized_path = [path[0]]
                for i in range(1, len(path)):
                    if self.is_line_clear(optimized_path[-1], path[i]):
                        continue  # 直接跳过，继续连线
                    optimized_path.append(path[i - 1])  # 断点处加入
                optimized_path.append(path[-1])  # 添加终点

                print(f"优化后的路径，共 {len(optimized_path)} 个点")
                return optimized_path

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
