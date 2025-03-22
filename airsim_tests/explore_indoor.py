import torch
import math
import numpy as np

from airsim_drone import SensorDroneController, PIDPathFollower, ImageProcessor, AStarPathfinder, select_landing_site
from airsim_drone.utils.voxel_grid import VoxelGridManager
from airsim_drone.visualization.visualize_planes_with_site import visualize_planes_on_image
from airsim_tests.grid_map_manager import GridMapManager


class AirSimDroneControllerTest(SensorDroneController):
    def __init__(self, ip):
        # 初始化体素网格管理器
        super().__init__(ip)
        self.voxel_manager = VoxelGridManager(voxel_size=0.1)
        self.astar_planner = AStarPathfinder(resolution=0.1, safety_margin=0.2)
        self.path_follower = PIDPathFollower(self.client)
        self.grid_manager = GridMapManager(grid_size=0.1)  # 2D 栅格地图管理器

    def rotate(self, angle_deg):
        """无人机旋转指定角度"""
        print(f"旋转 {angle_deg:.2f}°")
        _, _, current_yaw = self.path_follower.get_current_state()
        target_yaw = current_yaw + math.radians(angle_deg)
        self.path_follower.rotate_to_target_pid(target_yaw)

    def check_path_collision(self, start_pos, end_pos):
        """
        检查从 `start_pos` 到 `end_pos` 之间的连线是否穿过墙体
        :param start_pos: (x, y) 起点
        :param end_pos: (x, y) 终点
        :return: (bool, np.array) 是否穿墙 & 墙体法向量
        """
        x1, y1 = self.grid_manager.world_to_grid(*start_pos)
        x2, y2 = self.grid_manager.world_to_grid(*end_pos)

        # 计算 Bresenham 直线上的所有点
        line_points = self.grid_manager.bresenham_line(x1, y1, x2, y2)

        for point in line_points:
            if point in self.grid_manager.grid_map and self.grid_manager.grid_map[point] == 1:
                # 发现墙体，返回法向量
                wall_normal = self.grid_manager.grid_normals.get(point, np.array([1, 0]))  # 默认法向量 (1, 0)
                return True, wall_normal

        return False, None

    def forward_exploration(self, check_distance=2.5, move_distance=1.0):
        """向前探索，先检查路径是否穿墙，如有障碍则调整方向"""
        print("开始前向探索...")
        pos, _, yaw = self.path_follower.get_current_state()
        check_x = pos.x_val + check_distance * math.cos(yaw)
        check_y = pos.y_val + check_distance * math.sin(yaw)

        start_pos = (pos.x_val, pos.y_val)
        end_pos = (check_x, check_y)

        # **检查路径是否穿过墙体**
        collision, wall_normal = self.check_path_collision(start_pos, end_pos)

        if collision:
            print("路径穿越墙体，调整方向...")
            # **计算当前位置的朝向角度**
            current_yaw = math.degrees(yaw)
            # **计算墙体法向量的角度**
            wall_angle = math.degrees(math.atan2(wall_normal[1], wall_normal[0]))
            # **计算需要旋转的角度**
            turn_angle = (wall_angle - current_yaw + 90) % 360
            if turn_angle > 180:
                turn_angle -= 360  # 选择最短旋转方向
            self.rotate(turn_angle)
            return

        # **A* 规划路径**
        planned_path = self.astar_planner.search(self.voxel_manager, (pos.x_val, pos.y_val, pos.z_val), (check_x, check_y, pos.z_val))

        if planned_path:
            print(f"A* 路径规划成功，移动 {move_distance:.1f}m")
            move_x = pos.x_val + move_distance * math.cos(yaw)
            move_y = pos.y_val + move_distance * math.sin(yaw)
            move_z = pos.z_val
            move_path = self.astar_planner.search(self.voxel_manager, (pos.x_val, pos.y_val, pos.z_val), (move_x, move_y, move_z))

            if move_path:
                self.path_follower.move_along_path(self.voxel_manager, move_path)
        else:
            print("A* 规划失败，调整方向...")
            self.rotate(90)  # 如果无法规划路径，尝试旋转 180°


processor = ImageProcessor()
drone = AirSimDroneControllerTest(ip="172.21.74.24")
# 起飞
drone.takeoff(flight_height=1.5)
for i in range(30):
    drone.voxel_manager = VoxelGridManager(voxel_size=0.1)
    # 获取原图
    image, camera_position, camera_orientation = drone.get_image()
    # 深度估计与语义分割
    predictions, depth = processor.process_image(image)
    depth = depth * 0.74    # 根据测试，得到的较为合适的矫正系数
    # 转换点云
    points, valid_indices = drone.get_point_cloud(depth, camera_position, camera_orientation)
    # 使用管理类方法创建局部体素网格并合并到全局
    drone.voxel_manager.create_and_merge_local_map(points, visualize=False)
    # 语义筛选和平面提取
    allowed_planes, _, extracted_planes = processor.filter_and_extract_planes(drone.voxel_manager, predictions, points, valid_indices)

    # visualize_planes_on_image(image, allowed_planes, extracted_planes, save=False)

    drone_pos, drone_orientation = drone.get_drone_state()
    drone_pos = (drone_pos.x_val, drone_pos.y_val)
    drone.grid_manager.create_and_merge(extracted_planes, drone_pos, drone_orientation, drone.fov, visualize=True)

    drone.forward_exploration()

# 无人机降落
drone.land_and_release()
