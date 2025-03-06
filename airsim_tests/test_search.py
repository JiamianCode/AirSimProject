from airsim_drone import ImageProcessor, AirSimDroneController
from examples.Astar.visualize import visualize_3d_cloud


depth_model_path = '../depth_anything_v2_metric/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
sem_seg_config_file = '../MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml'
sem_seg_model_weights = '../MaskDINO/model/semantic_ade20k_48.7miou.pth'

processor = ImageProcessor(depth_model_path, sem_seg_config_file, sem_seg_model_weights)
# drone = AirSimDroneController(ip="172.19.72.21")
# drone = AirSimDroneController(ip="172.21.74.40")
drone = AirSimDroneController(ip="192.168.105.203")
# 起飞
drone.takeoff(flight_height=1.5)
# 获取原图
image, camera_position, camera_orientation = drone.get_image()
# 深度估计与语义分割
predictions, depth = processor.process_image(image)
# 转换点云
points, valid_indices = drone.get_point_cloud(depth, camera_position, camera_orientation)
# 语义筛选和平面提取
allowed_planes, unknown_planes = processor.filter_and_extract_planes(predictions, points, valid_indices)
# 可视化候选区
processor.visualize_planes_on_image(image, allowed_planes, unknown_planes)

# 可视化点云
# visualize_3d_cloud(points)
# 着陆
drone.land_and_release()
