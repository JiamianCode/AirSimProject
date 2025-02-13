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

    def reset(self):
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
    w = q.w_val
    x = q.x_val
    y = q.y_val
    z = q.z_val

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def move_to_target(client, pid_controllers, target_pos, target_yaw,
                  dt=0.1, pos_tol=0.1, yaw_tol=math.radians(2),
                  vel_tol=0.1, timeout=30):
    """
    改进版移动控制函数，增加速度稳定判断
    """
    start_time = time.time()
    for controller in pid_controllers['position'].values():
        controller.reset()
    for controller in pid_controllers['velocity'].values():
        controller.reset()
    pid_controllers['yaw'].reset()

    stable_counter = 0  # 速度稳定计数器
    required_stable = 5  # 需要连续5次稳定

    while True:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        _, _, current_yaw = quaternion_to_eulerian_angles(orientation)

        # 计算误差
        pos_error = {
            'x': target_pos.x_val - pos.x_val,
            'y': target_pos.y_val - pos.y_val,
            'z': target_pos.z_val - pos.z_val
        }
        yaw_error = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi

        # 位置环
        target_vel = {
            'x': pid_controllers['position']['x'].compute(pos_error['x'], dt),
            'y': pid_controllers['position']['y'].compute(pos_error['y'], dt),
            'z': pid_controllers['position']['z'].compute(pos_error['z'], dt)
        }

        # 速度环
        vel_error = {
            'x': target_vel['x'] - vel.x_val,
            'y': target_vel['y'] - vel.y_val,
            'z': target_vel['z'] - vel.z_val
        }
        acceleration = {
            'x': pid_controllers['velocity']['x'].compute(vel_error['x'], dt),
            'y': pid_controllers['velocity']['y'].compute(vel_error['y'], dt),
            'z': pid_controllers['velocity']['z'].compute(vel_error['z'], dt)
        }

        # 计算指令速度
        vx = vel.x_val + acceleration['x'] * dt
        vy = vel.y_val + acceleration['y'] * dt
        vz = vel.z_val + acceleration['z'] * dt

        # 偏航控制
        yaw_rate = pid_controllers['yaw'].compute(yaw_error, dt)

        # 发送指令
        client.moveByVelocityAsync(
            vx, vy, vz, dt,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))
        )

        # 检查稳定条件
        pos_distance = math.sqrt(sum([e**2 for e in pos_error.values()]))
        yaw_distance = abs(yaw_error)
        vel_magnitude = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)

        # 当接近目标时检查速度稳定性
        if pos_distance < pos_tol and yaw_distance < yaw_tol:
            if vel_magnitude < vel_tol:
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= required_stable:
                print(f"完全稳定到达目标！位置误差：{pos_distance:.2f}m，速度：{vel_magnitude:.2f}m/s")
                return True
        else:
            stable_counter = 0

        if time.time() - start_time > timeout:
            print("移动超时！最后状态：")
            print(f"位置误差：{pos_distance:.2f}m，速度：{vel_magnitude:.2f}m/s")
            return False

        time.sleep(dt)

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    pid_controllers = {
        'position': {
            'x': PIDController(0.5, 0.01, 0.2, 2.0, 3.0),
            'y': PIDController(0.5, 0.01, 0.2, 2.0, 3.0),
            'z': PIDController(1.0, 0.08, 0.2, 2.0, 3.0)
        },
        'velocity': {
            'x': PIDController(0.8, 0.0, 0.1, 1.0, 2.0),
            'y': PIDController(0.2, 0.0, 0.1, 1.0, 3.0),
            'z': PIDController(0.2, 0.0, 0.1, 1.0, 2.0)
        },
        'yaw': PIDController(0.5, 0.0, 0.1, 1.0, 1.0)
    }

    try:
        # === 测试用例1：起飞到-1.5m ===
        print("=== 测试1：起飞 ===")
        client.takeoffAsync().join()
        move_to_target(client, pid_controllers,
                      airsim.Vector3r(0, 0, -1.6), 0,
                      vel_tol=0.05)
        '''
        # === 测试2：飞到(3,-2,-1)偏航0° ===
        print("\n=== 测试2：移动到(3,-2,-1) ===")
        move_to_target(client, pid_controllers,
                      airsim.Vector3r(3, -2, -1), 0)

        
        # === 测试3：抬升到-1.5m ===
        print("\n=== 测试3：垂直抬升 ===")
        move_to_target(client, pid_controllers,
                      airsim.Vector3r(3, -2, -1.5), 0)

        # === 测试4：原地偏航45° ===
        print("\n=== 测试4：旋转到45° ===")
        move_to_target(client, pid_controllers,
                      airsim.Vector3r(3, -2, -1.5), math.radians(45))

        # === 测试5：悬停任务 ===
        print("\n=== 测试5：悬停2秒 ===")
        start = time.time()
        while time.time() - start < 2:
            # 持续获取当前状态并保持
            state = client.getMultirotorState()
            vel = state.kinematics_estimated.linear_velocity
            client.moveByVelocityAsync(
                vel.x_val, vel.y_val, vel.z_val, 0.1,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=False, yaw_or_rate=45)
            )
            time.sleep(0.1)
        
        # === 测试6：返回原点 ===
        print("\n=== 测试6：返回原点 ===")
        move_to_target(client, pid_controllers,
                      airsim.Vector3r(0, 0, -1), math.radians(-45))
        '''
    finally:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)

if __name__ == "__main__":
    main()