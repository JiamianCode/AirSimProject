from airsim_drone import SensorDroneController, PIDPathFollower, ImageProcessor
from airsim_tests.aStarPathfinder import AStarPathfinder
from airsim_tests.planning import evaluate_planes
from airsim_tests.voxel_grid import VoxelGridManager


class AirSimDroneControllerTest(SensorDroneController):
    def __init__(self, ip):
        # 初始化体素网格管理器
        super().__init__(ip)
        self.voxel_manager = VoxelGridManager(voxel_size=0.1)
        self.astar_planner = AStarPathfinder(resolution=0.1, safety_margin=0.2)
        self.path_follower = PIDPathFollower(self.client)

    def navigate_with_astar(self, start, goal):
        """
        使用 A* 进行路径规划并导航
        """
        path = self.astar_planner.search(self.voxel_manager, start, goal)  # 规划路径
        if path is None:
            print("A* 规划失败，无法到达目标点")
            return True

        print(f"初始路径有 ({len(path)} 个点)")
        # 优化路径并沿之移动
        state = self.path_follower.move_along_path(self.voxel_manager, path, use_optimization=True, visualize=False)
        if state:
            print("A* 导航完成，成功到达目标")

        return state


depth_model_path = '../depth_anything_v2_metric/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
sem_seg_config_file = '../MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml'
sem_seg_model_weights = '../MaskDINO/model/semantic_ade20k_48.7miou.pth'

processor = ImageProcessor(depth_model_path, sem_seg_config_file, sem_seg_model_weights)
drone = AirSimDroneControllerTest(ip="172.21.74.48")
# 起飞
drone.takeoff(flight_height=1.5)
while True:
    # 获取原图
    image, camera_position, camera_orientation = drone.get_image()
    # 深度估计与语义分割
    predictions, depth = processor.process_image(image)
    depth = depth * 0.74    # 根据测试，得到的较为合适的矫正系数
    # depth = processor.sharpen_real_depth(depth, threshold=200, dilate_size=5) # 深度分层，取消，改为侵蚀

    # 转换点云
    points, valid_indices = drone.get_point_cloud(depth, camera_position, camera_orientation)

    # 使用管理类方法创建局部体素网格并合并到全局
    drone.voxel_manager.create_and_merge_local_map(points)
    # drone.voxel_manager.visualize()

    # 语义筛选和平面提取
    allowed_planes, unknown_planes = processor.filter_and_extract_planes(drone.voxel_manager, predictions, points, valid_indices)
    # processor.visualize_planes_on_image(image, allowed_planes, unknown_planes)

    drone_pos, _ = drone.get_drone_state()
    drone_pos = (drone_pos.x_val, drone_pos.y_val, drone_pos.z_val)

    # 从 allowed_planes 中选出落点候选坐标
    sorted_centers_3d, sorted_centers_2d, sorted_scores = evaluate_planes(allowed_planes, drone_pos, k=1.5, visualize=False)

    # 可视化候选区
    processor.visualize_planes_on_image(image, allowed_planes, unknown_planes, sorted_centers_2d, sorted_scores)
    if sorted_centers_3d is None:
        break

    start_pos = drone_pos
    goal_pos = sorted_centers_3d[0]
    goal_pos = (goal_pos[0], goal_pos[1], goal_pos[2]-0.5)
    state = drone.navigate_with_astar(start_pos, goal_pos)  # 执行 A* 避障导航
    # print(drone.get_drone_state())

    if state:
        break

# 无人机降落
drone.land_and_release()
