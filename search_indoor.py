import numpy as np
from airsim import Vector3r

from airsim_drone import AirSimDroneController
from airsim_drone.process.depth_to_point_cloud import depth_to_point_cloud


class Voxel:
    def __init__(self, position, size):
        self.position = np.array(position)  # 体素中心坐标 (x,y,z)
        self.size = size  # 体素边长
        self.is_occupied = False  # 是否被占据
        self.is_explored = False  # 是否被探索过


class VoxelGrid:
    def __init__(self, voxel_size=0.5):
        self.voxel_size = voxel_size
        self.grid = {}  # {(x,y,z): Voxel}

    def world_to_voxel(self, point):
        """将世界坐标转换为体素坐标"""
        if isinstance(point, Vector3r):
            point = np.array([point.x_val, point.y_val, point.z_val])  # 处理 Vector3r 类型

        point = np.asarray(point, dtype=np.float64)  # 确保 point 是 numpy 数组
        return tuple((point // self.voxel_size).astype(int))  # 整数除法转换为体素索引

    def add_points(self, points):
        """将点云数据添加到体素网格"""
        for point in points:
            voxel_coord = self.world_to_voxel(point)
            if voxel_coord not in self.grid:
                voxel_center = (np.array(voxel_coord) * self.voxel_size + self.voxel_size / 2)
                self.grid[voxel_coord] = Voxel(voxel_center, self.voxel_size)
                self.grid[voxel_coord].is_occupied = True
                self.grid[voxel_coord].is_explored = True


class FrontierDetector:
    def __init__(self, voxel_grid):
        self.grid = voxel_grid

    def find_frontiers(self, current_pos, search_radius=5.0):
        """寻找前沿候选体素"""
        frontiers = []
        current_voxel = self.grid.world_to_voxel(current_pos)
        search_range = int(search_radius / self.grid.voxel_size)

        # 遍历周围体素
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                for dz in [-1, 0, 1]:
                    coord = (current_voxel[0] + dx, current_voxel[1] + dy, current_voxel[2] + dz)
                    if self._is_frontier(coord):
                        frontiers.append(self.grid.grid[coord].position)
        return frontiers

    def _is_frontier(self, coord):
        """判断是否为前沿体素"""
        if coord not in self.grid.grid:
            return False

        # 检查邻近是否存在未探索区域
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
                    if neighbor not in self.grid.grid:
                        return True
        return False


class ExplorationPlanner:
    def __init__(self, voxel_grid):
        self.grid = voxel_grid
        self.detector = FrontierDetector(voxel_grid)
        self.visited = set()

    def select_goal(self, current_pos):
        """选择下一个探索目标"""
        frontiers = self.detector.find_frontiers(current_pos)
        if not frontiers:
            return None, None

        # 筛选安全候选点
        safe_goals = []
        for pos in frontiers:
            if self._is_position_safe(pos) and self._is_visible(current_pos, pos):
                safe_goals.append(pos)

        if not safe_goals:
            return None, None

        # 选择最近目标并计算偏航角
        current_arr = np.array(current_pos)
        distances = np.linalg.norm(safe_goals - current_arr, axis=1)
        goal_pos = safe_goals[np.argmin(distances)]
        yaw = self._calculate_yaw(current_pos, goal_pos)

        return goal_pos, yaw

    def _is_position_safe(self, pos):
        """检查目标位置是否安全"""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_pos = pos + np.array([dx, dy, dz]) * self.grid.voxel_size
                    voxel = self.grid.world_to_voxel(check_pos)
                    if voxel in self.grid.grid and self.grid.grid[voxel].is_occupied:
                        return False
        return True

    def _is_visible(self, start_pos, target_pos):
        """简单视线检测"""
        direction = target_pos - start_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return True

        steps = int(distance / self.grid.voxel_size)
        for t in np.linspace(0, 1, steps):
            check_pos = start_pos + t * direction
            voxel = self.grid.world_to_voxel(check_pos)
            if voxel in self.grid.grid and self.grid.grid[voxel].is_occupied:
                return False
        return True

    def _calculate_yaw(self, current_pos, target_pos):
        """计算目标偏航角（弧度）"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        return np.arctan2(dy, dx)


def main():
    drone = AirSimDroneController()
    voxel_grid = VoxelGrid(voxel_size=0.5)
    planner = ExplorationPlanner(voxel_grid)

    try:
        drone.takeoff(flight_height=1.5)
        current_pos, _ = drone.get_drone_state()

        while True:
            # 获取深度图并生成点云
            _, depth_img, camera_position, camera_orientation = drone.get_images()
            points, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation)

            # 更新体素地图
            if len(points) > 0:
                voxel_grid.add_points(points)

            # 选择下一个目标
            goal_pos, goal_yaw = planner.select_goal(current_pos)
            if goal_pos is None:
                print("Exploration completed!")
                break

            print(f"Moving to: {goal_pos} with yaw: {np.degrees(goal_yaw):.1f}°")
            drone.move_to_position(goal_pos[0], goal_pos[1], current_pos[2], goal_yaw)
            current_pos, _ = drone.get_drone_state()

    finally:
        drone.land_and_release()


if __name__ == "__main__":
    main()