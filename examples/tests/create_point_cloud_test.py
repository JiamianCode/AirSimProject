import math

import airsim
import numpy as np

from airsim_drone import AirSimDroneController
from airsim_drone.controllers.point_cloud import depth_to_point_cloud
from airsim_drone.visualization.visualize import visualize_3d_cloud

# 初始化控制器
drone = AirSimDroneController()

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

# 获取全景点云
# point_cloud = drone.create_point_cloud()

all_points_world = []  # 存储初始扫描点云

x, y, z = 0, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_1, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = 0, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, math.pi / 2))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_2, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = 0, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, math.pi))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_3, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = 0, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, math.pi * 3 / 2))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_4, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = -3, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_11, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = -3, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, math.pi / 2))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_22, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = -3, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, math.pi))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_33, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

x, y, z = -3, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, math.pi * 3 / 2))
drone.client.simSetVehiclePose(pose, True)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_44, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)

'''
x, y, z = 0, 0, -1.5
pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(math.pi / 4, 0, 0))
drone.client.simSetCameraPose('', pose)
_, depth_img, camera_position, camera_orientation = drone.get_images()
point_cloud_5, _ = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                        max_depth=20.0)
'''
point_cloud = np.vstack((point_cloud_1, point_cloud_2, point_cloud_3, point_cloud_4,
                         point_cloud_11, point_cloud_22, point_cloud_33, point_cloud_44,
                         #point_cloud_11 + np.array([0, 0, 0]),
                         #point_cloud_22 + np.array([-3, 3, 0]),
                         #point_cloud_33 + np.array([-6, 0, 0]),
                         #point_cloud_44 + np.array([-3, -3, 0])
                         ))

# 可视化点云
visualize_3d_cloud(point_cloud)

# 降落并释放控制
drone.land_and_release()
