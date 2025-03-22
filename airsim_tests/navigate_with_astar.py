import time

from airsim_drone import SensorDroneController, PIDPathFollower, ImageProcessor, AStarPathfinder, select_landing_site
from airsim_drone.utils.voxel_grid import VoxelGridManager
from airsim_drone.visualization.visualize_planes_with_site import visualize_planes_on_image


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


processor = ImageProcessor()
drone = AirSimDroneControllerTest(ip="172.21.74.24")
# 起飞
drone.takeoff(flight_height=1.5)
while True:
    print("开始获取图像、分析图像、选择落点...")
    start_time = time.time()
    # 获取原图
    image, camera_position, camera_orientation = drone.get_image()
    # 深度估计与语义分割
    predictions, depth = processor.process_image(image)
    depth = depth * 0.74    # 根据测试，得到的较为合适的矫正系数
    # 转换点云
    points, valid_indices = drone.get_point_cloud(depth, camera_position, camera_orientation)
    # 使用管理类方法创建局部体素网格并合并到全局
    drone.voxel_manager.create_and_merge_local_map(points, visualize=True)
    # 语义筛选和平面提取
    allowed_planes, unknown_planes, _ = processor.filter_and_extract_planes(drone.voxel_manager, predictions, points, valid_indices)

    drone_pos, _ = drone.get_drone_state()
    drone_pos = (drone_pos.x_val, drone_pos.y_val, drone_pos.z_val)

    # 从 allowed_planes 中选出落点候选坐标
    sorted_centers_3d, sorted_centers_2d, sorted_scores = select_landing_site(allowed_planes, drone_pos, k=1.5, visualize=False)
    end_time = time.time()
    print(f"获取图像、分析图像、选择落点完成，运行时间: {round(end_time - start_time, 2)}秒")

    # 可视化候选区
    visualize_planes_on_image(image, allowed_planes, unknown_planes, sorted_centers_2d, sorted_scores,save=True)

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
