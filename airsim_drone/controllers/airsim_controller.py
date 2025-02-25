from .navigation_controller import NavigationDroneController
from .sensor_controller import SensorDroneController


class AirSimDroneController(NavigationDroneController, SensorDroneController):
    def __init__(self, ip="", vehicle_name=""):
        super().__init__(ip, vehicle_name)
