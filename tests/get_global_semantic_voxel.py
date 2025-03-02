import math

import airsim
import numpy as np

from examples.GetSemanticVoxel.voxel_grid import VoxelGridManager
from airsim_drone import LabelManager, SensorDroneController


class AirSimDroneControllerTest(SensorDroneController):
    def __init__(self):
        # 初始化体素网格管理器
        super().__init__()
        self.manager = VoxelGridManager(voxel_size=0.1)

    def can_move_forward(self, forward_distance):
        """根据网格判断是否可以向前飞行指定距离，考虑安全距离"""
        # 获取当前无人机的状态（位置和朝向）使用 get_drone_state
        current_position, current_orientation = self.get_drone_state()

        # 获取当前朝向（yaw角度，围绕Z轴旋转）
        yaw = airsim.to_eularian_angles(current_orientation)[2]  # 获取yaw角度

        # 根据yaw计算飞行方向：假设无人机的前方方向为yaw角度的方向
        forward_vector = airsim.Vector3r(forward_distance * math.cos(yaw),
                                         forward_distance * math.sin(yaw), 0)

        # 计算目标位置
        target_position = airsim.Vector3r(
            current_position.x_val + forward_vector.x_val,
            current_position.y_val + forward_vector.y_val,
            current_position.z_val + forward_vector.z_val
        )

        # 获取路径上各体素的网格（这里假设是一个简单的路径，沿直线插值）
        num_steps = int(forward_distance / self.manager.global_voxel_grid.voxel_size)  # 计算沿着路径的步数
        path_voxels = []

        for step in range(1, num_steps + 1):
            # 计算路径上每个点的坐标
            interpolated_position = np.array([
                current_position.x_val + step * forward_vector.x_val / num_steps,
                current_position.y_val + step * forward_vector.y_val / num_steps,
                current_position.z_val + step * forward_vector.z_val / num_steps
            ])
            voxel_key = self.manager.global_voxel_grid.get_voxel_key(interpolated_position)
            path_voxels.append(voxel_key)

        # 检查路径上的每个体素是否有障碍物并考虑安全距离
        for voxel_key in path_voxels:
            if voxel_key in self.manager.global_voxel_grid.grid:
                voxel = self.manager.global_voxel_grid.grid[voxel_key]
                # 如果该体素属于障碍物网格（假设障碍物为point_count > 0）
                if voxel.point_count > 0:
                    # 判断距离障碍物的距离，如果小于安全距离则不能前进
                    print(f"路径上发现障碍物，距离障碍物小于安全距离，无法前进：体素 {voxel_key}")
                    return False

        # 如果路径上没有障碍物或者障碍物与路径的距离大于安全距离，则可以前进
        print("路径没有障碍物或障碍物距离足够远，可以前进")
        return True

    def move_forward(self, forward_distance):
        """控制无人机向前飞行指定的距离"""
        print(f"向前飞行 {forward_distance} 米")
        current_position, current_orientation = self.get_drone_state()

        # 获取当前朝向（yaw角度，围绕Z轴旋转）
        yaw = airsim.to_eularian_angles(current_orientation)[2]

        # 根据yaw计算飞行方向
        forward_vector = airsim.Vector3r(forward_distance * math.cos(yaw),
                                         forward_distance * math.sin(yaw), 0)

        # 计算目标位置
        forward_position = airsim.Vector3r(
            current_position.x_val + forward_vector.x_val,
            current_position.y_val + forward_vector.y_val,
            current_position.z_val + forward_vector.z_val
        )

        # 控制无人机向前飞行
        self.client.moveToPositionAsync(forward_position.x_val, forward_position.y_val, forward_position.z_val,
                                        1).join()

    def rotate_clockwise(self, rotation_angle):
        """控制无人机顺时针旋转指定的角度"""
        print(f"顺时针旋转 {rotation_angle} 弧度")
        current_position, current_orientation = self.get_drone_state()

        # 获取当前朝向（yaw角度，围绕Z轴旋转）
        yaw = airsim.to_eularian_angles(current_orientation)[2]

        # 计算旋转后的目标姿态
        new_yaw = yaw + rotation_angle  # 顺时针旋转

        # 控制无人机旋转
        self.client.moveByRollPitchYawZAsync(0, 0, -new_yaw, -1.5, 1.5).join()

    def explore(self, threshold=100, forward_distance=2.0, rotation_angle=math.pi / 8, max_steps=50,
                max_rotation_attempts=5):
        """改进的环境探索策略"""
        step_count = 0  # 用于记录探索步数
        last_grid_change = 0  # 上一次网格变化量
        rotation_attempts = 0  # 用于记录旋转尝试次数

        while True:
            # 获取深度图像和语义分割图像
            semantic_img, depth_img, camera_position, camera_orientation = self.get_depth_and_semantic()
            # 获取点云数据
            points, valid_indices = self.get_point_cloud(depth_img, camera_position, camera_orientation)
            # 创建局部体素网格并合并到全局，获取网格变化量
            grid_change = self.manager.create_and_merge_local_map(points, valid_indices, candidate_labels,
                                                                  semantic_img, label_manager)

            # 判断网格变化量是否小于阈值，若小于则停止探索
            if grid_change < threshold and grid_change == last_grid_change:
                print("探索停止，网格变化量小于阈值且无变化")
                break

            last_grid_change = grid_change  # 更新上次网格变化量

            # 判断是否可以向前飞行
            if self.can_move_forward(forward_distance):
                self.move_forward(forward_distance)  # 向前移动一定距离
                step_count = 0  # 成功移动后重置步数
                rotation_attempts = 0  # 重置旋转尝试次数
            else:
                # 如果前方不能前进，尝试顺时针旋转一定角度
                self.rotate_clockwise(rotation_angle)  # 如果不能前进，则顺时针旋转

                step_count += 1
                rotation_attempts += 1

            # 检查是否达到最大探索步数限制
            if step_count >= max_steps:
                print("最大探索步数达到，尝试改变方向或增加探索范围")
                # 重新评估旋转角度，增加更大幅度的旋转
                rotation_angle += math.pi / 8  # 增加旋转角度尝试新的方向
                print(f"增加旋转角度，当前旋转角度: {rotation_angle}")
                step_count = 0  # 重置步数


# 初始化标签管理器
label_manager = LabelManager('../airsim_drone/utils/object_labels.csv')  # 标签文件路径
# 候选语义标签
candidate_labels = ['floor', 'table', 'chair', 'carpet']

# 初始化无人机控制器
drone = AirSimDroneControllerTest()

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

# 开始环境探索
print("开始环境探索")
drone.explore(threshold=100, forward_distance=2.0, rotation_angle=math.pi / 2)

# 可视化全局体素网格
drone.manager.visualize(True)

# 无人机降落
drone.land_and_release()
