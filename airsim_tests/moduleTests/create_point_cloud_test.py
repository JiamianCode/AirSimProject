import math

import airsim
import cv2
import numpy as np

from airsim_drone import SensorDroneController
from airsim_drone.process.depth_to_point_cloud import depth_to_point_cloud
from examples.Astar.visualize import visualize_3d_cloud


class AirSimDroneControllerTest(SensorDroneController):
    def get_images(self, camera_name="front_center"):
        """
        统一获取 RGB 图像和深度图，以及相机的位置和姿态
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
        ], vehicle_name=self.vehicle_name)

        if len(responses) < 2:
            raise RuntimeError("无法同时获取 RGB 和 Depth 图像")

        img_bgr_resp = responses[0]
        img_depth_resp = responses[1]

        img_bgr = np.frombuffer(img_bgr_resp.image_data_uint8, dtype=np.uint8).reshape(img_bgr_resp.height,
                                                                                       img_bgr_resp.width, 3)
        depth_img = np.array(img_depth_resp.image_data_float, dtype=np.float32).reshape(img_depth_resp.height,
                                                                                        img_depth_resp.width)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR到RGB的转换

        # 获取相机的全局位置和姿态信息
        camera_position = img_bgr_resp.camera_position
        camera_orientation = img_bgr_resp.camera_orientation

        return img_rgb, depth_img, camera_position, camera_orientation


# 初始化控制器
drone = AirSimDroneControllerTest()

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
                         # point_cloud_11 + np.array([0, 0, 0]),
                         # point_cloud_22 + np.array([-3, 3, 0]),
                         # point_cloud_33 + np.array([-6, 0, 0]),
                         # point_cloud_44 + np.array([-3, -3, 0])
                         ))

# 可视化点云
visualize_3d_cloud(point_cloud)

# 降落并释放控制
drone.land_and_release()
