# 深度图转世界坐标
import numpy as np
from scipy.spatial.transform import Rotation


def depth_to_point_cloud(drone, depth_img, camera_position, camera_orientation, max_depth=20.0):
    # 过滤无效深度
    valid_mask = (depth_img > 0) & (depth_img <= max_depth) & (~np.isnan(depth_img))
    if not np.any(valid_mask):
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)

    h, w = depth_img.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    u_valid, v_valid, d_valid = uu[valid_mask], vv[valid_mask], depth_img[valid_mask]

    # 获取内参
    K = drone.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 像素转相机坐标
    X_c = (u_valid - cx) * d_valid / fx
    Y_c = (v_valid - cy) * d_valid / fy
    Z_c = d_valid
    points_camera = np.vstack([X_c, Y_c, Z_c]).T

    # 相机坐标系 映射 机体坐标系
    R_camera_to_body = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    drone_pos = np.array([camera_position.x_val, camera_position.y_val, camera_position.z_val])
    points_body = (R_camera_to_body @ points_camera.T).T + drone_pos

    # 机体坐标系 转换 世界坐标系
    rot = Rotation.from_quat([camera_orientation.x_val, camera_orientation.y_val,
                              camera_orientation.z_val, camera_orientation.w_val])
    R_body_to_world = rot.as_matrix()

    points_world = (R_body_to_world @ points_body.T).T

    valid_indices = np.vstack([u_valid, v_valid]).T

    print(f"点云数量: {len(points_world)}")

    # 相机坐标系:东、地、北 --映射--> 无人机机体坐标系:北、东、地 --转换--> 无人机世界坐标系：北、东、地
    # 可以进一步在可视化时( --映射--> 可视化世界坐标系)
    return points_world, valid_indices
