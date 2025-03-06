import airsim
import numpy as np

# 与 airsim 建立连接
client = airsim.MultirotorClient()
client.confirmConnection()

# 确定是否要用API控制
client.enableApiControl(True)

# 解锁无人机转起来
client.armDisarm(True)

# join()等任务结束再进行下个任务
# 起飞
# client.takeoffAsync().join()

# 飞行
# client.moveToZAsync(-1.5, 1).join()  # 飞到3m高
# client.moveToPositionAsync(3, 0, -1.5, 1).join()
# client.moveToPositionAsync(3, -1, -1.5, 1).join()
# client.moveToPositionAsync(0, -1, -1.5, 1).join()
# client.moveToPositionAsync(0, 0, -1.5, 1).join()

with open('reference_point.csv', 'r') as f:
    lines = f.readlines()
reference_x = float(lines[0])
reference_y = float(lines[1])
print("Landing at: (", reference_x, reference_y, ")")

client.moveToPositionAsync(reference_x, -reference_y, 0, 1).join()

# 降落
client.landAsync().join()

# 上锁
client.armDisarm(False)
# 释放控制权
client.enableApiControl(False)

print("Mission completed!")