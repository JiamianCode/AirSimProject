import airsim


class BaseDroneController:
    def __init__(self, vehicle_name=""):
        """
        初始化无人机控制器
        """
        self.client = airsim.MultirotorClient()
        self.vehicle_name = vehicle_name
        self.client.confirmConnection()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        print("已连接到 AirSim 并获取控制权限")

    def takeoff(self, flight_height=1.5):
        """
        控制无人机起飞
        """
        print(f"起飞中，目标高度: {flight_height} m")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(-flight_height, 1, vehicle_name=self.vehicle_name).join()
        print("无人机已起飞")

    def land_and_release(self):
        """
        控制无人机降落并释放控制
        """
        print("无人机正在降落...")
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)
        print("任务结束，已降落并解除控制")

    def get_drone_state(self):
        """
        获取无人机的当前位置和姿态
        """
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        return position, orientation
