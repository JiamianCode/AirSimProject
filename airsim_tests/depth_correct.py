import airsim
import numpy as np
from airsim_drone import ImageProcessor, SensorDroneController


class AirSimDroneControllerTest(SensorDroneController):
    def __init__(self, ip):
        # 初始化体素网格管理器
        super().__init__(ip)

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

        # 获取相机的全局位置和姿态信息
        camera_position = img_bgr_resp.camera_position
        camera_orientation = img_bgr_resp.camera_orientation

        return img_bgr, depth_img, camera_position, camera_orientation


depth_model_path = '../depth_anything_v2_metric/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
sem_seg_config_file = '../MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml'
sem_seg_model_weights = '../MaskDINO/model/semantic_ade20k_48.7miou.pth'

processor = ImageProcessor(depth_model_path, sem_seg_config_file, sem_seg_model_weights)
drone = AirSimDroneControllerTest(ip="172.21.74.48")
# 起飞
drone.takeoff(flight_height=1.5)
# 获取原图
image, camera_position, camera_orientation = drone.get_image()
# 深度估计与语义分割
predictions, depth = processor.process_image(image)
_, depth_img,_ ,_ = drone.get_images()

"""
                    要对着非窗户的位置进行矫正，否则窗户外面存在较大偏差！
        测得4个位置：
            尺度因子: 0.7673, 矫正前误差: 0.7203, 矫正后误差: 0.2029
            尺度因子: 0.7480, 矫正前误差: 1.2030, 矫正后误差: 0.1712
            尺度因子: 0.7374, 矫正前误差: 1.3747, 矫正后误差: 0.1482
            尺度因子: 0.7250, 矫正前误差: 0.7793, 矫正后误差: 0.0658
"""

# 假设 depth 和 depth_img 都是 NumPy 数组，并且尺寸相同
depth = depth.flatten()  # 展平为 1D 数组
depth_img = depth_img.flatten()  # 真实深度，同样展平

# 计算尺度校正因子 lambda
lambda_factor = np.sum(depth_img * depth) / np.sum(depth ** 2)

# 计算修正后的深度
depth_corrected = lambda_factor * depth

# 计算误差
error_before = np.mean(np.abs(depth - depth_img))
error_after = np.mean(np.abs(depth_corrected - depth_img))

print(f"尺度因子: {lambda_factor:.4f}")
print(f"矫正前误差: {error_before:.4f}, 矫正后误差: {error_after:.4f}")

# 着陆
drone.land_and_release()
