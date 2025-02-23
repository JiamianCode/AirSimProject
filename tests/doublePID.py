import airsim
import time
import math


class PIDController:
    def __init__(self, kp, ki, kd, max_integral, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(min(output, self.max_output), -self.max_output)

        self.prev_error = error
        return output


def quaternion_to_eulerian_angles(q):
    """将四元数转换为欧拉角（roll, pitch, yaw）"""
    w = q.w_val
    x = q.x_val
    y = q.y_val
    z = q.z_val

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# 连接 AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 初始化目标参数
target_position = airsim.Vector3r(3, -2, -1)  # 目标位置（X, Y, Z）
target_yaw = math.radians(45)  # 目标偏航角（45度）

# 初始化PID控制器
# 位置环（X/Y/Z）
pos_x_pid = PIDController(kp=0.5, ki=0.01, kd=0.2, max_integral=2.0, max_output=3.0)
pos_y_pid = PIDController(kp=0.5, ki=0.01, kd=0.2, max_integral=2.0, max_output=3.0)
pos_z_pid = PIDController(kp=0.5, ki=0.01, kd=0.2, max_integral=2.0, max_output=3.0)

# 速度环（X/Y/Z）
vel_x_pid = PIDController(kp=0.8, ki=0.0, kd=0.1, max_integral=1.0, max_output=2.0)
vel_y_pid = PIDController(kp=0.8, ki=0.0, kd=0.1, max_integral=1.0, max_output=2.0)
vel_z_pid = PIDController(kp=0.8, ki=0.0, kd=0.1, max_integral=1.0, max_output=2.0)

# 航向角控制环
yaw_pid = PIDController(kp=0.5, ki=0.0, kd=0.1, max_integral=1.0, max_output=1.0)

# 控制参数
dt = 0.1  # 控制周期
position_tolerance = 0.2  # 位置容差（米）
yaw_tolerance = math.radians(2)  # 偏航角容差（2度）

# 起飞到初始高度
client.takeoffAsync().join()
client.moveToZAsync(target_position.z_val, 3).join()

while True:
    # 获取无人机状态
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    velocity = state.kinematics_estimated.linear_velocity
    orientation = state.kinematics_estimated.orientation

    # 计算欧拉角（获取当前偏航角）
    roll, pitch, yaw = quaternion_to_eulerian_angles(orientation)

    # ========== 位置控制环 ==========
    # 计算位置误差
    error_x = target_position.x_val - position.x_val
    error_y = target_position.y_val - position.y_val
    error_z = target_position.z_val - position.z_val

    # 位置环计算目标速度
    target_vx = pos_x_pid.compute(error_x, dt)
    target_vy = pos_y_pid.compute(error_y, dt)
    target_vz = pos_z_pid.compute(error_z, dt)

    # ========== 速度控制环 ==========
    # 计算速度误差
    error_vx = target_vx - velocity.x_val
    error_vy = target_vy - velocity.y_val
    error_vz = target_vz - velocity.z_val

    # 速度环计算加速度指令
    ax = vel_x_pid.compute(error_vx, dt)
    ay = vel_y_pid.compute(error_vy, dt)
    az = vel_z_pid.compute(error_vz, dt)

    # 计算速度指令
    vx_cmd = velocity.x_val + ax * dt
    vy_cmd = velocity.y_val + ay * dt
    vz_cmd = velocity.z_val + az * dt

    # ========== 航向角控制 ==========
    # 计算偏航角误差（考虑角度环绕）
    error_yaw = target_yaw - yaw
    error_yaw = (error_yaw + math.pi) % (2 * math.pi) - math.pi

    # 计算偏航角速率指令
    yaw_rate_cmd = yaw_pid.compute(error_yaw, dt)

    # 构造Yaw控制模式
    yaw_mode = airsim.YawMode(
        is_rate=True,
        yaw_or_rate=math.degrees(yaw_rate_cmd)  # AirSim使用度数单位
    )

    # 发送综合控制指令
    client.moveByVelocityAsync(
        vx_cmd,
        vy_cmd,
        vz_cmd,
        dt,
        airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode
    )

    # 检查是否到达目标
    pos_distance = math.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)
    yaw_distance = abs(error_yaw)

    if pos_distance < position_tolerance and yaw_distance < yaw_tolerance:
        print("到达目标位置和朝向！")
        break

    time.sleep(dt)

# 降落并断开连接
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)