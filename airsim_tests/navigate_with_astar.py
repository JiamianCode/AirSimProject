from airsim_drone import SensorDroneController, AStarPathfinder, PIDPathFollower, ImageProcessor
from airsim_tests.voxel_grid import VoxelGridManager


class AirSimDroneControllerTest(SensorDroneController):
    def __init__(self, ip):
        # 初始化体素网格管理器
        super().__init__(ip)
        self.voxel_manager = VoxelGridManager(voxel_size=0.1)
        self.astar_planner = AStarPathfinder(resolution=0.1, safety_margin=0.2)
        self.path_follower = PIDPathFollower(self.client)

    def create_occupancy_grid(self):
        """
        从 VoxelGridManager 生成占用地图
        """
        self.astar_planner.set_occupancy_grid(self.voxel_manager)
        print("占用栅格地图已建立")

    def navigate_with_astar(self, start, goal):
        """
        使用 A* 进行路径规划并导航
        """
        self.create_occupancy_grid()  # 生成占用地图
        path = self.astar_planner.search(start, goal)  # 规划路径

        if path is None:
            print("A* 规划失败，无法到达目标点")
            return

        print(f"初始路径有 ({len(path)} 个点)")
        # 优化路径并沿之移动
        self.path_follower.move_along_path(path, use_optimization=True)

        print("A* 导航完成，成功到达目标")


depth_model_path = '../depth_anything_v2_metric/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
sem_seg_config_file = '../MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml'
sem_seg_model_weights = '../MaskDINO/model/semantic_ade20k_48.7miou.pth'

processor = ImageProcessor(depth_model_path, sem_seg_config_file, sem_seg_model_weights)
# drone = AirSimDroneControllerTest(ip="172.19.72.21")
# drone = AirSimDroneControllerTest(ip="172.21.74.40")
drone = AirSimDroneControllerTest(ip="192.168.105.203")
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

# 使用管理类方法创建局部体素网格并合并到全局
drone.voxel_manager.create_and_merge_local_map(points)
# 可视化全局体素网格
drone.voxel_manager.visualize()

start_pos = (0, 0, -1.5)
goal_pos = (7, -1, -1.5)

drone.navigate_with_astar(start_pos, goal_pos)  # 执行 A* 避障导航
print(drone.get_drone_state())

# 无人机降落
drone.land_and_release()
