import airsim
import cv2
import numpy as np

from examples.GetSemanticVoxel.voxel_grid import VoxelGridManager
from airsim_drone import SensorDroneController, LabelManager


class AirSimDroneControllerTest(SensorDroneController):
    def flytest(self, yaw):
        # self.client.moveToPositionAsync(2.5, 0, -1.5, 1).join()
        self.client.moveByRollPitchYawZAsync(0, 0, yaw, -1.5, 1.5,
                                             vehicle_name=self.vehicle_name).join()

    def get_depth_and_semantic(self, camera_name="front_center"):
        # 请求深度图像和语义分割图像
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False)
        ], vehicle_name=self.vehicle_name)

        img_depth_resp = responses[0]
        img_bgr_resp = responses[1]

        img_bgr = np.frombuffer(img_bgr_resp.image_data_uint8, dtype=np.uint8).reshape(
            img_bgr_resp.height, img_bgr_resp.width, 3)
        depth_img = np.array(img_depth_resp.image_data_float, dtype=np.float32).reshape(
            img_depth_resp.height, img_depth_resp.width)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        camera_position = img_bgr_resp.camera_position
        camera_orientation = img_bgr_resp.camera_orientation

        return img_rgb, depth_img, camera_position, camera_orientation

    # 可视化掩码的函数
    def visualize_mask_on_image(self, semantic_img, label_manager, candidate_labels):
        # 获取原图
        img_rgb, _, _ = self.get_image()
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 遍历每个候选标签
        for label in candidate_labels:
            object_ids = label_manager.label_to_ids.get(label, [])

            # 创建掩码图像
            mask = np.zeros(semantic_img.shape[:2], dtype=np.uint8)  # 使用二维形状初始化掩码
            for object_id in object_ids:
                # 生成掩码：检查每个像素是否属于当前标签
                mask = np.all(semantic_img == object_id, axis=-1)
                mask = mask.astype(np.uint8) * 255  # 转为二值化掩码
                color = tuple(object_id)  # 使用 object_id 直接作为 RGB 颜色

                # 创建与原图大小相同的空图像
                mask_colored = np.zeros_like(img_rgb)
                mask_colored[mask == 255] = color  # 只有掩码区域才会被赋予颜色

                # 透明叠加掩码到原图，透明度设置为 0.5
                img_rgb = cv2.addWeighted(img_rgb, 1.0, mask_colored, 0.5, 0)

        # 显示叠加后的图像
        cv2.imshow("Semantic Mask Visualization", img_rgb)
        cv2.imwrite("../../output/SemanticMask.png", img_rgb)
        # 等待用户按键退出
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 初始化标签管理器
label_manager = LabelManager('../../airsim_drone/utils/object_labels.csv')  # 标签文件路径
# 候选语义标签
candidate_labels = ['floor', 'table', 'chair']

# 初始化管理类
manager = VoxelGridManager(voxel_size=0.1)

# 初始化无人机控制器
drone = AirSimDroneControllerTest()

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

# 获取深度图像和语义分割图像
semantic_img, depth_img, camera_position, camera_orientation = drone.get_depth_and_semantic()

# 获取点云数据
points, valid_indices = drone.get_point_cloud(depth_img, camera_position, camera_orientation)

# 使用管理类方法创建局部体素网格并合并到全局
manager.create_and_merge_local_map(points, valid_indices, candidate_labels, semantic_img, label_manager)

# 可视化全局体素网格
manager.visualize()

# 可视化掩码图像并显示
drone.visualize_mask_on_image(semantic_img, label_manager, candidate_labels)

# 降落无人机
drone.land_and_release()
