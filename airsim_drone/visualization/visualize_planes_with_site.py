import datetime
import os

import cv2
import numpy as np


def visualize_planes_on_image(image, allowed_planes, unknown_planes, sorted_centers_2d=None, sorted_scores=None, save=True):
    """
    可视化平面掩码，并在原始图像上叠加类别标签。
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    image_output = image_rgb.copy()  # 创建一个副本用于显示叠加结果

    # 定义颜色和透明度（BGR 格式）
    allowed_color = (0, 255, 0)    # 绿色 (允许平面)
    unknown_color = (0, 0, 255)    # 红色 (未知平面)
    alpha = 0.5                    # 半透明度

    # 绘制允许的平面
    for plane in allowed_planes:
        mask = plane['mask']  # 平面掩码
        category = plane['category']  # 平面类别
        # 创建半透明掩码
        transparent_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        transparent_overlay[mask.cpu().numpy() == 1] = allowed_color
        # 将掩码叠加到图像副本上
        image_output = cv2.addWeighted(image_output, 1, transparent_overlay, alpha, 0)

        # 标注类别名称（掩码的中间位置）
        coords = np.column_stack(np.where(mask.cpu().numpy() == 1))
        if len(coords) > 0:
            center = np.mean(coords, axis=0).astype(int)  # 计算掩码区域的中心位置
            center = (center[1], center[0])  # 调整为 (x, y) 格式
            cv2.putText(image_output, category, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # 绘制未知的平面
    for plane in unknown_planes:
        mask = plane['mask']  # 平面掩码
        category = plane['category']  # 平面类别
        # 创建半透明掩码（背景色）
        transparent_overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        transparent_overlay[mask.cpu().numpy() == 1] = unknown_color
        # 将掩码叠加到图像副本上
        image_output = cv2.addWeighted(image_output, 1, transparent_overlay, alpha, 0)

        # 计算掩码区域的中心位置并添加文本
        coords = np.column_stack(np.where(mask.cpu().numpy() == 1))
        if len(coords) > 0:
            center = np.mean(coords, axis=0).astype(int)  # 计算掩码区域的中心位置
            center = (center[1], center[0])  # 调整为 (x, y) 格式
            cv2.putText(image_output, category, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 绘制 sorted_centers_2d
    if sorted_centers_2d:
        num_centers = len(sorted_centers_2d)
        for i, (u, v) in enumerate(sorted_centers_2d):
            red_intensity = 255 - int((i / max(1, num_centers - 1)) * 180)  # 颜色渐变
            center_color = (0, 0, red_intensity)

            # 画圆点
            cv2.circle(image_output, (u, v), 5, center_color, -1)

            # 标注编号 (1, 2, 3...)
            cv2.putText(image_output, f"{i+1}", (u+10, v+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # 标注得分
            if sorted_scores:
                cv2.putText(image_output, f"{sorted_scores[i]:.2f}", (u+10, v+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 是否存储为文件
    if save:
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../../output/fly_path/visualized_planes_{timestamp}.png"
        # 保存图像
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.abspath(os.path.join(script_dir, filename))

        cv2.imwrite(absolute_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))
    else:
        # 创建可调整大小的窗口并显示结果
        cv2.namedWindow("Visualized Planes", cv2.WINDOW_NORMAL)  # 设置窗口为可调大小
        cv2.imshow("Visualized Planes", cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))  # 转回 BGR 格式显示
        cv2.waitKey(0)
        cv2.destroyAllWindows()
