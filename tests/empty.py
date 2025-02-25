import time

import airsim
import cv2
import numpy as np
from airsim_drone import SensorDroneController
from examples.moduleTests.LabelManager import LabelManager


class AirSimDroneControllerTest(SensorDroneController):
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


# 初始化无人机控制器
drone = AirSimDroneControllerTest()

# 初始化标签管理器
label_manager = LabelManager('../examples/moduleTests/object_labels.csv')  # 标签文件路径

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

# 获取深度图像和语义分割图像
start_time = time.time()
semantic_img, depth_img, camera_position, camera_orientation = drone.get_depth_and_semantic()
end_time = time.time()
print(f"获取图像时间: {round(end_time - start_time, 2)}秒")

# 获取点云数据
points, valid_indices = drone.get_point_cloud(depth_img, camera_position, camera_orientation)

# 候选语义标签
candidate_labels = ['floor', 'table', 'chair']


# 对点云坐标进行语义赋值的函数
def assign_semantic_to_points(points, valid_indices, candidate_labels, label_manager):
    # 存储每个点云坐标对应的标签
    semantic_values = []

    # 遍历每个有效的像素索引
    for idx in range(valid_indices.shape[0]):
        u, v = valid_indices[idx]
        # 获取当前像素对应的点云坐标
        point = points[idx]

        # 获取当前像素的标签
        object_id = tuple(semantic_img[v, u])

        # 如果该物体属于候选标签之一，则赋值标签，否则为None
        label = None
        for candidate_label in candidate_labels:
            if object_id in label_manager.label_to_ids.get(candidate_label, []):
                label = candidate_label
                break

        # 将标签添加到结果列表中
        semantic_values.append((point, label))

    return semantic_values


# 可视化掩码的函数
def visualize_mask_on_image(semantic_img, img_rgb, label_manager, candidate_labels):
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 为每个候选标签分配不同的颜色
    label_colors = {
        'floor': (0, 255, 0),  # 绿色
        'table': (0, 0, 255),  # 红色
        'chair': (255, 0, 0)  # 蓝色
    }

    # 遍历每个候选标签
    for label in candidate_labels:
        object_ids = label_manager.label_to_ids.get(label, [])

        # 创建掩码图像
        mask = np.zeros(semantic_img.shape, dtype=np.uint8)
        for object_id in object_ids:
            # 生成掩码：检查每个像素是否属于当前标签
            mask = np.all(semantic_img == object_id, axis=-1)
            mask = mask.astype(np.uint8) * 255  # 转为二值化掩码

            # 为了可视化效果，在原图上叠加颜色
            color = label_colors.get(label, (255, 255, 255))  # 默认颜色为白色
            mask_colored = np.zeros_like(img_rgb)
            mask_colored[mask == 255] = color

            # 透明叠加掩码到原图，透明度设置为0.5
            img_rgb = cv2.addWeighted(img_rgb, 1.0, mask_colored, 0.5, 0)

    # 显示叠加后的图像
    cv2.imshow("Semantic Mask Visualization", img_rgb)
    # 等待用户按键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 对点云进行语义赋值
semantic_values = assign_semantic_to_points(points, valid_indices, candidate_labels, label_manager)

# 可视化掩码图像并显示
img_rgb, _, _ = drone.get_image()
visualize_mask_on_image(semantic_img, img_rgb, label_manager, candidate_labels)

# 降落无人机
drone.land_and_release()
