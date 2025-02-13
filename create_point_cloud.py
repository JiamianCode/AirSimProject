from src.drone_control import AirSimDroneController
from tools.visualize import visualize_3d_cloud

# 初始化控制器
drone = AirSimDroneController()

# 起飞至1.5米高度
drone.takeoff(flight_height=1.5)

# 获取全景点云
point_cloud = drone.create_point_cloud()

# 可视化点云
visualize_3d_cloud(point_cloud)

# 降落并释放控制
drone.land_and_release()
