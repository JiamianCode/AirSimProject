import airsim
import matplotlib
import numpy as np
import cv2
import time
import math
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import sys

matplotlib.use('TkAgg')


################################################################################
# 1) 无人机连接、起飞、深度图采集与转换等基础函数
################################################################################

def connect_and_takeoff(client, flight_height=2.0, vehicle_name=""):
    """
    连接到AirSim并控制无人机起飞。
    """
    print("连接到AirSim...")
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)

    print(f"起飞中，目标高度: {flight_height} m")
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    # AirSim中Z轴向下为负数，这里moveToZAsync填入负值
    client.moveToZAsync(-flight_height, 1, vehicle_name=vehicle_name).join()
    print("无人机已起飞")


def get_IntrinsicMatrix(client, camera_name, vehicle_name=""):
    """
    根据相机的FOV和图像尺寸，计算内参矩阵(K)。
    """
    # 获取相机信息
    camera_info = client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name, external=False)
    fov = camera_info.fov  # 水平FOV

    # 请求一张图像来获取分辨率
    response = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ], vehicle_name=vehicle_name)

    width = response[0].width
    height = response[0].height

    # 计算焦距 (fx, fy) = (cx/tan(fov/2), same)
    # 此处假设像素方形: fx == fy
    fx = width / 2 / math.tan(math.radians(fov / 2))
    fy = fx

    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def acquire_data(client, camera_name="front_center", vehicle_name="", max_depth=20.0):
    """
    从指定相机获取深度图(DepthPlanar)和无人机当前姿态，返回 (depth_img, position, orientation)。
    """
    # 获取深度图
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
    ], vehicle_name=vehicle_name)
    if len(responses) == 0:
        raise RuntimeError("无法从AirSim获取深度图")

    depth_response = responses[0]
    depth_img = np.array(depth_response.image_data_float, dtype=np.float32)
    depth_img = depth_img.reshape(depth_response.height, depth_response.width)

    # 获取无人机姿态
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    position = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation

    return depth_img, position, orientation


def process_depth_data(client, depth_img, position, orientation,
                       camera_name="front_center",
                       vehicle_name="",
                       max_depth=20.0):
    """
    将深度图转换为世界坐标系下的3D点云。
    返回点云 (N,3) 的numpy数组。
    """
    # 过滤无效深度
    valid_mask = (depth_img > 0) & (depth_img <= max_depth) & (~np.isnan(depth_img))
    if not np.any(valid_mask):
        return np.empty((0, 3))

    h, w = depth_img.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    u_valid = uu[valid_mask]
    v_valid = vv[valid_mask]
    d_valid = depth_img[valid_mask]

    # 获取内参
    K = get_IntrinsicMatrix(client, camera_name, vehicle_name)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 像素转相机坐标
    X_c = (u_valid - cx) * d_valid / fx
    Y_c = (v_valid - cy) * d_valid / fy
    Z_c = d_valid
    points_camera = np.vstack([X_c, Y_c, Z_c]).T  # (N,3)

    # 相机坐标 -> 世界坐标
    rot = Rotation.from_quat([orientation.x_val,
                              orientation.y_val,
                              orientation.z_val,
                              orientation.w_val])
    R = rot.as_matrix()
    points_world = (R @ points_camera.T).T

    # 坐标系校正
    R_inv = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    points_world = (R_inv @ points_world.T).T

    # 减去无人机的平移
    drone_pos = np.array([position.x_val, position.y_val, position.z_val])
    points_world = points_world - drone_pos

    return points_world


################################################################################
# 2) 鼠标点击获取目标像素 -> 转3D坐标
################################################################################

mouse_click_pos = None
window_name = "AirSim RGB View"


def mouse_callback(event, x, y, flags, param):
    """
    OpenCV鼠标回调，捕捉左键点击坐标 (x, y).
    x, y 对应图像上(列, 行).
    """
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)
        print(f"鼠标点击像素 = ({x}, {y})")


def get_goal_from_click(client, camera_name="front_center", vehicle_name="", max_depth=20.0):
    """
    获取当前RGB/深度图，显示给用户鼠标点击 -> 返回点击处的3D世界坐标 (goal_world)。
    如果点击处深度无效，返回 None。
    """
    # 1. 获取RGB & 深度图
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
    ], vehicle_name=vehicle_name)
    if len(responses) < 2:
        raise RuntimeError("无法同时获取RGB和Depth图像")

    img_rgb_resp = responses[0]
    img_depth_resp = responses[1]
    w = img_rgb_resp.width
    h = img_rgb_resp.height

    # 转numpy
    img_rgb_1d = np.frombuffer(img_rgb_resp.image_data_uint8, dtype=np.uint8)
    if w * h * 3 != len(img_rgb_1d):
        raise ValueError("RGB数据不匹配，无法reshape")
    img_rgb = img_rgb_1d.reshape(h, w, 3)

    depth_raw = np.array(img_depth_resp.image_data_float, dtype=np.float32).reshape(h, w)

    # 2. 打开OpenCV窗口让用户点击
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w // 2, h // 2)
    cv2.setMouseCallback(window_name, mouse_callback)

    global mouse_click_pos
    mouse_click_pos = None

    print("请在弹出的窗口中点击目标位置(按 ESC 取消选择)...")
    while True:
        cv2.imshow(window_name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("用户按ESC取消选择。")
            break
        if mouse_click_pos is not None:
            # 已点击
            break
    cv2.destroyAllWindows()

    if mouse_click_pos is None:
        return None  # 用户未点击

    # 3. 计算点击处的3D坐标
    (u, v) = mouse_click_pos
    depth_value = depth_raw[v, u]
    if depth_value <= 0 or depth_value > max_depth or np.isnan(depth_value):
        print(f"点击({u},{v})深度无效: {depth_value}.")
        return None

    # 内参
    K = get_IntrinsicMatrix(client, camera_name, vehicle_name)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X_c = (u - cx) * depth_value / fx
    Y_c = (v - cy) * depth_value / fy
    Z_c = depth_value
    point_camera = np.array([X_c, Y_c, Z_c])

    # 无人机姿态
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    orientation = state.kinematics_estimated.orientation
    position = state.kinematics_estimated.position

    rot = Rotation.from_quat([orientation.x_val,
                              orientation.y_val,
                              orientation.z_val,
                              orientation.w_val])
    R = rot.as_matrix()
    point_world = R.dot(point_camera)

    R_inv = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    point_world = R_inv.dot(point_world)
    drone_pos = np.array([position.x_val, position.y_val, position.z_val])
    point_world = point_world - drone_pos

    point_world[2] += 0.15

    print(f"点击像素({u},{v}) => 深度 = {depth_value:.3f} => 世界坐标 = {point_world}")
    return point_world


################################################################################
# 3) 三维占用栅格构建 + A*搜索 + 可视化
################################################################################

def create_3d_occupancy_grid(points_world, resolution=0.1, safety_margin=0.1):
    """
    根据点云构建三维占用栅格。
    points_world: shape(N,3)
    resolution: 每个格子的大小 (m)
    safety_margin: 障碍扩张距离 (m)
    返回:
      occupancy_grid: bool数组, shape=(nx, ny, nz), True表示障碍
      origin: (x_min, y_min, z_min)
      shape: (nx, ny, nz)
    """
    if len(points_world) == 0:
        raise ValueError("点云为空，无法构建栅格地图")

    x_min = np.min(points_world[:, 0])
    x_max = np.max(points_world[:, 0])
    y_min = np.min(points_world[:, 1])
    y_max = np.max(points_world[:, 1])
    z_min = np.min(points_world[:, 2])
    z_max = np.max(points_world[:, 2])

    # 稍微留余量
    margin = 0.3
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    z_min -= margin
    z_max += margin

    nx = int(math.ceil((x_max - x_min) / resolution))
    ny = int(math.ceil((y_max - y_min) / resolution))
    nz = int(math.ceil((z_max - z_min) / resolution))

    print(f"3D栅格范围 X:[{x_min:.2f},{x_max:.2f}], Y:[{y_min:.2f},{y_max:.2f}], Z:[{z_min:.2f},{z_max:.2f}]")
    print(f"3D栅格大小: nx={nx}, ny={ny}, nz={nz} (每格{resolution}m)")

    occupancy_grid = np.zeros((nx, ny, nz), dtype=bool)

    # 安全距离对应的格子半径
    margin_cells = int(math.ceil(safety_margin / resolution))

    for (x, y, z) in points_world:
        ix = int((x - x_min) // resolution)
        iy = int((y - y_min) // resolution)
        iz = int((z - z_min) // resolution)
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                for dz in range(-margin_cells, margin_cells + 1):
                    nx_ = ix + dx
                    ny_ = iy + dy
                    nz_ = iz + dz
                    if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                        occupancy_grid[nx_, ny_, nz_] = True

    return occupancy_grid, (x_min, y_min, z_min), (nx, ny, nz)


def world_to_grid(x, y, z, origin, resolution):
    """
    世界坐标 -> 栅格坐标
    origin=(x_min, y_min, z_min)
    """
    (x_min, y_min, z_min) = origin
    ix = int((x - x_min) // resolution)
    iy = int((y - y_min) // resolution)
    iz = int((z - z_min) // resolution)
    return (ix, iy, iz)


def grid_to_world(ix, iy, iz, origin, resolution):
    """
    栅格坐标 -> 世界坐标 (此处简单以格心为对应位置)
    """
    (x_min, y_min, z_min) = origin
    x = ix * resolution + x_min + resolution / 2.0
    y = iy * resolution + y_min + resolution / 2.0
    z = iz * resolution + z_min + resolution / 2.0
    return (x, y, z)


def a_star_3d(occupancy_grid, start_cell, goal_cell):
    """
    三维A*搜索。
    occupancy_grid: shape=(nx, ny, nz), True表示障碍
    start_cell, goal_cell: (ix, iy, iz) 起止点栅格坐标
    返回: 路径(网格坐标列表), 若无解返回空列表
    """
    nx, ny, nz = occupancy_grid.shape

    def is_valid(ix, iy, iz):
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            return not occupancy_grid[ix, iy, iz]
        return False

    def heuristic(a, b):
        (ax, ay, az) = a
        (bx, by, bz) = b
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    open_set = []
    heapq.heappush(open_set, (0, start_cell))
    came_from = {}
    g_score = {start_cell: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_cell:
            # 回溯路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_cell)
            path.reverse()
            return path

        cx, cy, cz = current
        for dx, dy, dz in neighbors_6:
            nx_ = cx + dx
            ny_ = cy + dy
            nz_ = cz + dz
            if not is_valid(nx_, ny_, nz_):
                continue
            cost = g_score[current] + 1.0  # 简单+1
            if (nx_, ny_, nz_) not in g_score or cost < g_score[(nx_, ny_, nz_)]:
                g_score[(nx_, ny_, nz_)] = cost
                f_score = cost + heuristic((nx_, ny_, nz_), goal_cell)
                heapq.heappush(open_set, (f_score, (nx_, ny_, nz_)))
                came_from[(nx_, ny_, nz_)] = current

    return []


def visualize_3d_path(points_world, path_cells, origin, resolution, start_cell, goal_cell):
    """
    可视化三维点云，并标注路径、起点和终点。
    points_world: (N,3) numpy数组，表示点云
    path_cells: A* 规划得到的路径 (栅格坐标)
    origin: 栅格坐标系的原点 (x_min, y_min, z_min)
    resolution: 栅格的分辨率
    start_cell: 起点的栅格坐标
    goal_cell: 终点的栅格坐标
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 1) 绘制点云
    X = points_world[:, 0]
    Y = points_world[:, 1]
    Z = points_world[:, 2]
    sc = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=3, alpha=0.8, label='Point Cloud')

    # 2) 绘制路径
    if path_cells:
        path_world = [grid_to_world(ix, iy, iz, origin, resolution) for (ix, iy, iz) in path_cells]
        px = [p[0] for p in path_world]
        py = [p[1] for p in path_world]
        pz = [p[2] for p in path_world]
        ax.plot(px, py, pz, c='blue', linewidth=2, label='Path')
    else:
        print("无可行路径，无法可视化路径线段")

    # 3) 绘制起点和终点
    sx, sy, sz = grid_to_world(*start_cell, origin, resolution)
    gx, gy, gz = grid_to_world(*goal_cell, origin, resolution)
    ax.scatter(sx, sy, sz, c='green', s=80, marker='o', label='Start')
    ax.scatter(gx, gy, gz, c='red', s=80, marker='x', label='Goal')

    # 颜色条
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, aspect=10, shrink=0.5)
    cbar.set_label('Z (Elevation)', rotation=15, labelpad=15)

    # 轴标签
    ax.set_xlabel('X', labelpad=12)
    ax.set_ylabel('Y', labelpad=12)
    ax.set_zlabel('Z', labelpad=12)
    ax.view_init(elev=10, azim=-170)

    plt.title("3D Point Cloud with Path")
    plt.legend()
    plt.tight_layout()
    plt.show()


################################################################################
# 4) 主流程
################################################################################

def main():
    """
    1. 连接并起飞
    2. 获取点云
    3. 鼠标点击获取目标落点
    4. 建立3D栅格，执行A*
    5. 可视化占用栅格、路径等
    """
    client = airsim.MultirotorClient()
    flight_height = 1.5
    camera_name = "front_center"
    vehicle_name = ""

    try:
        # 1) 启动并起飞
        connect_and_takeoff(client, flight_height, vehicle_name)

        # 2) 获取深度图 -> 点云
        depth_img, drone_pos, drone_orient = acquire_data(client, camera_name, vehicle_name)
        points_world = process_depth_data(client, depth_img, drone_pos, drone_orient,
                                          camera_name, vehicle_name, max_depth=20.0)
        print(f"点云数量: {len(points_world)}")

        if len(points_world) == 0:
            print("未能生成有效点云，结束。")
            return

        # 3) 鼠标点击获取目标落点
        goal_world = get_goal_from_click(client, camera_name, vehicle_name, max_depth=20.0)
        if goal_world is None:
            print("未选择或无效深度，无法规划。结束。")
            return

        # 起点：假设无人机当前位置(即(0,0,flight_height)在点云坐标系中)。
        # 这里为了示例方便，直接用 (0,0,drone_pos.z_val)
        # 如果想更精确，就把无人机世界坐标(0,0,z)真正换算到处理后的点云坐标系。
        start_world = np.array([1.1, 0.0, -drone_pos.z_val])
        print(f"规划起点(世界坐标) = {start_world}")
        print(f"规划终点(世界坐标) = {goal_world}")

        # 4) 构建3D占用栅格 & A*搜索
        resolution = 0.1
        safety_margin = 0.1

        occupancy_grid, origin, (nx, ny, nz) = create_3d_occupancy_grid(points_world,
                                                                        resolution,
                                                                        safety_margin)

        # 起止点转栅格坐标
        start_cell = world_to_grid(start_world[0], start_world[1], start_world[2], origin, resolution)
        goal_cell = world_to_grid(goal_world[0], goal_world[1], goal_world[2], origin, resolution)

        # 判断起点/终点是否在范围内或已被占用
        if (start_cell[0] < 0 or start_cell[0] >= nx or
                start_cell[1] < 0 or start_cell[1] >= ny or
                start_cell[2] < 0 or start_cell[2] >= nz or
                occupancy_grid[start_cell[0], start_cell[1], start_cell[2]]):
            print("起点在障碍或超出栅格范围，无法规划。")
            return
        if (goal_cell[0] < 0 or goal_cell[0] >= nx or
                goal_cell[1] < 0 or goal_cell[1] >= ny or
                goal_cell[2] < 0 or goal_cell[2] >= nz or
                occupancy_grid[goal_cell[0], goal_cell[1], goal_cell[2]]):
            print("终点在障碍或超出栅格范围，无法规划。")
            return

        print(f"起点栅格坐标: {start_cell}, 终点栅格坐标: {goal_cell}")

        path_cells = a_star_3d(occupancy_grid, start_cell, goal_cell)
        if not path_cells:
            print("A*未找到可行路径。")
        else:
            print(f"成功找到路径，长度(包含起终点) = {len(path_cells)}")

        # 5) 可视化三维栅格、路径
        visualize_3d_path(points_world, path_cells, origin, resolution, start_cell, goal_cell)

        # 6) 控制无人机沿路径飞行并降落
        print("开始沿路径飞行...")
        path_world = [grid_to_world(ix, iy, iz, origin, resolution) for (ix, iy, iz) in path_cells]

        for waypoint in path_world:
            x, y, z = waypoint
            client.moveToPositionAsync(x, -y, -z, velocity=1.5, vehicle_name=vehicle_name).join()
            # time.sleep(0.1)  # 等待片刻，确保移动平稳

        print("到达目标点，准备降落...")
        client.landAsync(vehicle_name=vehicle_name).join()
        print("降落完成，任务结束。")

    except Exception as e:
        print(f"异常: {e}")
    finally:
        # 降落&解锁
        client.landAsync(vehicle_name=vehicle_name).join()
        client.armDisarm(False, vehicle_name)
        client.enableApiControl(False, vehicle_name)
        print("任务结束，已降落并解除控制。")


if __name__ == "__main__":
    main()
