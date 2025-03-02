import math

import airsim
import cv2
import numpy as np

from airsim_drone import depth_to_point_cloud
from airsim_drone.controllers.sensor_controller import SensorDroneController
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
    def get_image_and_point_could(self):
        """
        获取深度图并生成点云
        """
        img_rgb, depth_img, camera_position, camera_orientation = self.get_images()
        points, valid_indices = depth_to_point_cloud(self, depth_img, camera_position, camera_orientation)
        return img_rgb, points, valid_indices

    def create_point_cloud(self):
        """
        无人机在当前位置采集全局点云
        1. 无人机原地旋转 360° 采集点云数据
        2. 寻找空旷区域，计算下一个采样位置（暂未实现）
        3. 飞向目标点
        4. 计算飞行方向，再次原地旋转 360° 采集点云数据
        5. 返回原位置和姿态
        """

        def capture_360_point_cloud(yaw_start, z_height):
            """
            让无人机在当前位置进行 360° 旋转，并采集点云数据
            """
            all_points = []
            print(f"开始 360° 旋转，采集点云...")
            for i in range(4):  # 旋转 4 次，每次 90°
                yaw_start += math.radians(90)
                self.client.moveByRollPitchYawZAsync(0, 0, yaw_start, z_height, 1.5,
                                                     vehicle_name=self.vehicle_name).join()

                # 采集图像并转换为点云
                _, points_world, _ = self.get_image_and_point_could()

                if points_world is not None and points_world.size > 0:
                    all_points.append(points_world)

            if all_points:
                return np.vstack(all_points)
            else:
                print("点云数据为空")
                return None

        # 初始采集点云
        position, orientation = self.get_drone_state()
        yaw_start = airsim.to_eularian_angles(orientation)[2]  # 初始 Yaw 角

        first_points_world = capture_360_point_cloud(yaw_start, position.z_val)

        '''
        # 寻找目标空旷区域
        print("寻找空旷区域...")
        target_position = self.find_open_space(initial_point_cloud, position, min_dist=3.0, max_dist=4.0)

        if target_position is None:
            print("未找到合适的空旷区域，任务中止")
            return None

        target_x, target_y, target_z = target_position
        '''
        # 暂时使用该方法实现，往x轴正方向飞行3m
        target_x, target_y, target_z = position.x_val + 3, 0, position.z_val

        # 飞往目标点
        print(f"飞往空旷位置: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        self.client.moveToPositionAsync(target_x, target_y, target_z, velocity=1.5,
                                        vehicle_name=self.vehicle_name).join()

        # 第二次采集点云
        second_points_world = capture_360_point_cloud(yaw_start, target_z)

        # 6. 返回原位置和姿态
        print("返回原始位置并恢复朝向")
        self.client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, velocity=1.5,
                                        vehicle_name=self.vehicle_name).join()
        self.client.moveByRollPitchYawZAsync(0, 0, yaw_start, position.z_val, 1.5,
                                             vehicle_name=self.vehicle_name).join()
        print("任务完成，返回初始位置和姿态")

        return np.vstack((first_points_world, second_points_world))

    def find_open_space(self, point_cloud, current_position, min_dist=3.0, max_dist=4.0, height_tolerance=0.2):
        """
        在无人机当前高度的平面上，寻找空旷的位置，并返回目标点坐标
        """
        # return target_x, target_y, target_z


# 初始化控制器
drone = AirSimDroneControllerTest()

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

# 获取全景点云
point_cloud = drone.create_point_cloud()

# 可视化点云
visualize_3d_cloud(point_cloud)

# 降落并释放控制
drone.land_and_release()
