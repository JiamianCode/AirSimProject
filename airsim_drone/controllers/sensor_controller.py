import math
import airsim
import numpy as np

from .base_controller import BaseDroneController
from airsim_drone.process.depth_to_point_cloud import depth_to_point_cloud


class SensorDroneController(BaseDroneController):
    def __init__(self, ip="", vehicle_name=""):
        super().__init__(ip, vehicle_name)
        self.K = None
        self.get_image()
        print('camera is normal')

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

    def get_image(self, camera_name="front_center"):
        """
        获取 RGB 图像，检查并初始化相机矩阵
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
        ], vehicle_name=self.vehicle_name)

        img_bgr_resp = responses[0]

        img_bgr = np.frombuffer(img_bgr_resp.image_data_uint8, dtype=np.uint8).reshape(img_bgr_resp.height,
                                                                                       img_bgr_resp.width, 3)

        # 获取相机的全局位置和姿态信息
        camera_position = img_bgr_resp.camera_position
        camera_orientation = img_bgr_resp.camera_orientation

        # 初始化相机矩阵
        if self.K is None:
            # 获取相机的视场角 (fov)
            fov = self.client.simGetCameraInfo(camera_name, vehicle_name=self.vehicle_name).fov
            width = img_bgr_resp.width
            height = img_bgr_resp.height
            self.K = self.get_intrinsic_matrix(width, height, fov)

        return img_bgr, camera_position, camera_orientation

    def get_depth_and_semantic(self, camera_name="front_center"):
        """
        统一获取 深度图 和 语义分割图像，以及相机的位置和姿态
        """
        # 请求深度图像和语义分割图像
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False)
        ], vehicle_name=self.vehicle_name)

        img_depth_resp = responses[0]
        img_seg_resp = responses[1]

        img_seg = np.frombuffer(img_seg_resp.image_data_uint8, dtype=np.uint8).reshape(img_seg_resp.height,
                                                                                       img_seg_resp.width, 3)
        img_depth = np.array(img_depth_resp.image_data_float, dtype=np.float32).reshape(img_depth_resp.height,
                                                                                        img_depth_resp.width)

        camera_position = img_seg_resp.camera_position
        camera_orientation = img_seg_resp.camera_orientation

        return img_seg, img_depth, camera_position, camera_orientation

    def get_point_cloud(self, depth, camera_position, camera_orientation):
        """
        传入深度图并生成点云
        """
        points, valid_indices = depth_to_point_cloud(self, depth, camera_position, camera_orientation)
        return points, valid_indices
