import csv

import airsim
import cv2
import numpy as np

from airsim_drone import SensorDroneController


class AirSimDroneControllerTest(SensorDroneController):
    def get_semantic_segmentation(self, camera_name="front_center"):
        """
        获取语义分割图像
        """
        # 请求语义分割图像
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False),
        ], vehicle_name=self.vehicle_name)

        img_seg_resp = responses[0]
        img_seg = np.frombuffer(img_seg_resp.image_data_uint8, dtype=np.uint8).reshape(
            img_seg_resp.height, img_seg_resp.width, 3
        )

        # 多此一举，删去
        # img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)

        '''
        # 在图像上显示物体ID
        h, w, _ = img_seg.shape
        object_ids = set()
        for i in range(h):
            for j in range(w):
                # 获取当前像素的物体ID（每个像素的值即为对应物体的ID）
                object_id = tuple(img_seg[i, j])
                object_ids.add(object_id)  # 将物体 ID 添加到集合中（自动去重）
    
        print(len(object_ids))
        for id in object_ids:
            print(id)
    
        target_id = [tuple(i) for i in [ 
            (195, 176, 115),
            (217, 54, 65),
        ]]
    
        # 创建一个空白图像，默认设置为黑色（空区域）
        output_img = np.zeros_like(img_seg)
    
        # 只保留目标 ID 的区域，其余区域为黑色
        for i in range(img_seg.shape[0]):  # 遍历图像的每一行
            for j in range(img_seg.shape[1]):  # 遍历图像的每一列
                # 获取当前像素的物体 ID
                object_id = tuple(img_seg[i, j])
    
                # 如果该像素的 ID 等于目标 ID，则保留颜色，否则设置为空（黑色）
                if object_id in target_id:
                    output_img[i, j] = img_seg[i, j]
                else:
                    output_img[i, j] = [0, 0, 0]  # 设置为空，黑色
        '''

        # 获取语义分割结果
        # 每个像素的值表示该位置的类别标签
        return img_seg, img_seg_resp.camera_position, img_seg_resp.camera_orientation


# 用于保存物体ID和对应字符串的列表
id_string_list = []

# 初始化控制器
drone = AirSimDroneControllerTest()

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

semantic_img, camera_position, camera_orientation = drone.get_semantic_segmentation()


# 定义鼠标回调函数，用于处理鼠标点击事件
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 当鼠标左键点击时触发
        # 获取鼠标位置的像素值（即物体的ID）
        object_id = semantic_img[y, x]

        # 打印物体的ID
        print(f"Mouse clicked at position ({x}, {y}) with Object ID: {object_id}")

        # 允许用户输入字符串
        user_input = input(f"Enter a label for Object ID {object_id} (Press Enter to skip): ").strip()

        # 如果用户输入了内容，则保存到列表中
        if user_input:
            id_string_list.append((object_id, user_input))
            print(f"Saved: ({object_id}, {user_input})")
        else:
            print("No label entered, skipped.")


# 设置鼠标回调
cv2.namedWindow(winname='show')
cv2.setMouseCallback("show", on_mouse_click)

while True:
    cv2.imshow('show', semantic_img)
    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
# 显示结果
cv2.imshow("Semantic Segmentation", semantic_img)
cv2.waitKey(0)
'''
cv2.destroyAllWindows()

# 将结果保存到 CSV 文件
with open('../output/object_labels.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Label"])  # 写入标题
    for obj_id, label in id_string_list:
        writer.writerow([obj_id, label])  # 写入数据

print("Saved labels to 'object_labels.csv'.")

# 降落并释放控制
drone.land_and_release()
