import math
import time

import airsim
import cv2
import numpy as np

from src.depth_to_point_cloud import depth_to_point_cloud


class AirSimDroneController:
    def __init__(self, vehicle_name=""):
        """
        初始化无人机控制器
        """
        self.client = airsim.MultirotorClient()
        self.vehicle_name = vehicle_name
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        print("已连接到 AirSim 并获取控制权限")

        self.K = None

    def takeoff(self, flight_height=1.5):
        """
        控制无人机起飞
        """
        print(f"起飞中，目标高度: {flight_height} m")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(-flight_height, 1, vehicle_name=self.vehicle_name).join()
        print("无人机已起飞")

    def land_and_release(self):
        """
        控制无人机降落并释放控制
        """
        print("无人机正在降落...")
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)
        print("任务结束，已降落并解除控制")

    def get_drone_state(self):
        """
        获取无人机的当前位置和姿态
        """
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        return position, orientation

    @staticmethod
    def get_intrinsic_matrix(width, height, fov):
        """
        获取相机内参矩阵 K
        """
        # 计算焦距 (fx, fy) = (cx/tan(fov/2), same)，假设像素方形: fx == fy
        fx = width / 2 / math.tan(math.radians(fov / 2))
        fy = fx
        cx = width / 2
        cy = height / 2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K

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

        # 获取相机的视场角 (fov)
        fov = self.client.simGetCameraInfo(camera_name, vehicle_name=self.vehicle_name).fov
        width = img_bgr_resp.width
        height = img_bgr_resp.height

        self.K = self.get_intrinsic_matrix(width, height, fov)

        return img_rgb, depth_img, camera_position, camera_orientation

    def navigate_path(self, path_world):
        print("开始沿路径飞行...")
        for waypoint in path_world:
            x, y, z = waypoint
            self.client.moveToPositionAsync(x, y, z, velocity=1.5, vehicle_name=self.vehicle_name).join()
            time.sleep(0.1)  # 等待片刻，确保移动平稳
        print("到达目标点")

    def create_point_cloud(self):

        position, orientation = self.get_drone_state()
        yaw_start = airsim.to_eularian_angles(orientation)[2]  # 初始 Yaw 角
        fixed_z = position.z_val  # 记录初始高度
        all_points_world = []  # 存储初始扫描点云

        print("开始原地旋转，采集点云...")
        for i in range(4):  # 旋转 4 次，每次 90°
            yaw_start += math.radians(90)
            self.client.moveByRollPitchYawZAsync(0, 0, yaw_start, fixed_z, 1.5,
                                                 vehicle_name=self.vehicle_name).join()
            time.sleep(0.5)

            # 采集图像并转换为点云
            rgb_img, depth_img, camera_position, camera_orientation = self.get_images()
            points_world, _ = depth_to_point_cloud(self, depth_img, camera_position, camera_orientation, max_depth=20.0)

            if points_world is not None and points_world.size > 0:
                all_points_world.append(points_world)

        return np.vstack(all_points_world)
