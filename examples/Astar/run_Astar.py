import numpy as np

from airsim_drone.controllers.point_cloud import depth_to_point_cloud
from examples.Astar.a_star import AStar3D
from airsim_drone import AirSimDroneController
from examples.Astar.landing_select import get_goal_from_click
from airsim_drone.visualization.visualize import visualize_3d_path


def main():
    drone = AirSimDroneController()

    try:
        # 启动并起飞
        drone.takeoff(flight_height=1.5)

        # 获取彩色图像和深度图
        rgb_img, depth_img, camera_position, camera_orientation = drone.get_images()

        # 深度图->点云
        points_world, valid_indices = depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation,
                                                           max_depth=20.0)

        # 鼠标点击获取目标落点
        goal_world = get_goal_from_click(rgb_img, depth_img, points_world, valid_indices)

        # 起点：无人机当前位置
        position, _ = drone.get_drone_state()
        start_world = np.array([position.x_val, position.y_val, position.z_val])
        print(f"规划起点(世界坐标) = {start_world},\n规划终点(世界坐标) = {goal_world}")

        # 构建3D占用栅格 并且使用 A*搜索
        astar = AStar3D(resolution=0.1, safety_margin=0.1)
        astar.create_occupancy_grid(points_world)
        path_world = astar.search(start_world, goal_world)

        # 可视化三维栅格、路径
        visualize_3d_path(points_world, path_world, start_world, goal_world)

        # 控制无人机沿路径飞行
        drone.navigate_path(path_world)

    except Exception as e:
        print(f"异常: {e}")
    finally:
        # 无人机降落并解除控制
        drone.land_and_release()


if __name__ == "__main__":
    main()
