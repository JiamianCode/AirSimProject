import airsim
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

matplotlib.use('TkAgg')


# 计算相机的内参矩阵
def get_IntrinsicMatrix(client, camera_name, vehicle_name=""):
    intrinsic_matrix = np.zeros([3, 3])

    # 获取相机的视场角 (fov)
    fov = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name, external=False).fov

    # 请求图像数据以获取图像尺寸
    request = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
    responses = client.simGetImages(request, vehicle_name=vehicle_name)
    img_width = responses[0].width
    img_height = responses[0].height

    # 根据视场角和图像尺寸计算内参矩阵
    intrinsic_matrix[0, 0] = img_width / 2 / math.tan(math.radians(fov / 2))
    intrinsic_matrix[1, 1] = img_width / 2 / math.tan(math.radians(fov / 2))
    intrinsic_matrix[0, 2] = img_width / 2
    intrinsic_matrix[1, 2] = img_height / 2
    intrinsic_matrix[2, 2] = 1

    return intrinsic_matrix


# 连接并起飞无人机
def connect_and_takeoff(client, flight_height):
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 起飞到指定高度
    client.takeoffAsync().join()
    client.moveToZAsync(-flight_height, 1).join()


# 获取真实深度图和无人机状态数据
def acquire_data(client):
    responses = client.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.DepthPlanar, True, False)
    ])
    depth_img = np.array(responses[0].image_data_float).reshape(
        responses[0].height, responses[0].width)
    return depth_img, responses[0].camera_position, responses[0].camera_orientation


# 处理深度图数据并转换到世界坐标系
def process_depth_data(client, depth_img, position, orientation, max_depth):
    # 过滤无效深度值
    valid_mask = (depth_img > 0) & (depth_img <= max_depth) & ~np.isnan(depth_img)

    # 生成像素坐标
    height, width = depth_img.shape
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    u_valid = uu[valid_mask]
    v_valid = vv[valid_mask]
    d_valid = depth_img[valid_mask]

    # 相机坐标系转换
    intrinsic = get_IntrinsicMatrix(client, "front_center")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    X_c = (u_valid - cx) * d_valid / fx
    Y_c = (v_valid - cy) * d_valid / fy
    Z_c = d_valid
    points_camera = np.vstack([X_c, Y_c, Z_c]).T

    # 相机坐标系映射机体坐标系
    R_camera_to_body = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    drone_pos = np.array([position.x_val, position.y_val, position.z_val])
    points_body = (R_camera_to_body @ points_camera.T).T

    # 机体坐标系转换世界坐标系
    rot = Rotation.from_quat([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
    R_body_to_world = rot.as_matrix()

    points_world = (R_body_to_world @ points_body.T).T + drone_pos

    # 相机坐标系:东、地、北 --映射--> 无人机机体坐标系:北、东、地 --转换--> 无人机世界坐标系：北、东、地 --映射--> 可视化世界坐标系
    return points_world


# 3D散点可视化
def visualize_3d_elevation(points):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    R_world_to_show = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    points = (R_world_to_show @ points.T).T

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # 绘制散点，颜色随 Z 轴变化
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8)

    # 设置颜色条
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, aspect=10, shrink=0.5)
    cbar.set_label('Z (Elevation)', rotation=15, labelpad=15)

    # 设置坐标轴标签
    ax.set_xlabel('X', labelpad=12)
    ax.set_ylabel('Y', labelpad=12)
    ax.set_zlabel('Z', labelpad=12)

    # 设置视角
    ax.view_init(elev=10, azim=-170)

    plt.title("3D Point Cloud - Elevation Mapping")
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    client = airsim.MultirotorClient()
    flight_height = 1.5
    max_depth = 20.0

    try:
        # 启动无人机
        connect_and_takeoff(client, flight_height)

        # 数据采集
        depth_img, pos, orient = acquire_data(client)

        # 数据处理
        points_world = process_depth_data(client, depth_img, pos, orient, max_depth)
        print(f"有效点数: {len(points_world)}")

        # 可视化
        visualize_3d_elevation(points_world)

    except Exception as e:
        print(f"运行错误: {str(e)}")
    finally:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("任务终止，已安全降落")


if __name__ == "__main__":
    main()
