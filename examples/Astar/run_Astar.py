import airsim
import cv2
import numpy as np

from examples.Astar.a_star import AStar3D
from airsim_drone import depth_to_point_cloud, SensorDroneController, NavigationDroneController
from examples.Astar.landing_select import get_goal_from_click
from examples.Astar.visualize import visualize_3d_path


class AirSimDroneControllerTest(SensorDroneController, NavigationDroneController):
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


def main():
    drone = AirSimDroneControllerTest()

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
