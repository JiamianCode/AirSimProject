import cv2
import numpy as np

mouse_click_pos = None


def mouse_callback(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)
        print(f"鼠标点击像素 = ({x}, {y})")


def get_goal_from_click(img_rgb, depth_img, points_world, valid_indices):
    window_name = "AirSim RGB View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img_rgb.shape[1] // 2, img_rgb.shape[0] // 2)
    cv2.setMouseCallback(window_name, mouse_callback)

    global mouse_click_pos
    mouse_click_pos = None

    print("请在弹出的窗口中点击目标位置")
    while True:
        cv2.imshow(window_name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        if mouse_click_pos is not None:
            break  # 已点击
    cv2.destroyAllWindows()

    if mouse_click_pos is None:
        return None  # 用户未点击

    u, v = mouse_click_pos
    depth_value = depth_img[v, u]

    if depth_value <= 0 or np.isnan(depth_value):
        print(f"点击 ({u},{v}) 深度无效: {depth_value}.")
        return None

    idx = np.where((valid_indices[:, 0] == u) & (valid_indices[:, 1] == v))[0]

    if len(idx) == 0:
        print(f"点击 ({u},{v}) 不在有效点云范围内。")
        return None

    goal_world = points_world[idx[0]].copy()  # 非常重要！否则会改变原点云中的点位置，导致目标点必定处在障碍区内

    print(f"点击像素 ({u},{v}) => 深度 = {depth_value:.3f} => 世界坐标 = {goal_world}")
    return goal_world
