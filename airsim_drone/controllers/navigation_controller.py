import time
import numpy as np
from .base_controller import BaseDroneController

class NavigationDroneController(BaseDroneController):
    def navigate_path(self, path_world):
        """
        沿路径飞行
        """
        print("开始沿路径飞行...")
        for waypoint in path_world:
            x, y, z = waypoint
            self.client.moveToPositionAsync(x, y, z, velocity=1.5, vehicle_name=self.vehicle_name).join()
            time.sleep(0.1)  # 等待片刻，确保移动平稳
        print("到达目标点")

    # temporary method
    def move_to_position(self, x, y, z, yaw=None, velocity=2.0):
        """
        移动到指定位置
        """
        if yaw is not None:
            self.client.rotateToYawAsync(np.degrees(yaw), timeout_sec=3).join()
        self.client.moveToPositionAsync(x, y, -z, velocity).join()