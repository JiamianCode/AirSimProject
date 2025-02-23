import airsim

# 与 airsim 建立连接
client = airsim.MultirotorClient()
client.confirmConnection()

# 确定是否要用API控制
client.enableApiControl(True)

# 解锁无人机转起来
client.armDisarm(True)

# join()等任务结束再进行下个任务
# 起飞
client.takeoffAsync().join()

# 飞行
client.moveToZAsync(-1.5, 1).join()  # 飞到3m高
client.moveToPositionAsync(3, 0, -1.5, 1).join()
client.moveToPositionAsync(3, -1, -1.5, 1).join()
# client.moveToPositionAsync(0, -1, -1.5, 1).join()
client.moveToPositionAsync(0, 0, -1.5, 1).join()

# 降落
client.landAsync().join()

# 上锁
client.armDisarm(False)
# 释放控制权
client.enableApiControl(False)