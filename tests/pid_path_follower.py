import airsim
import time
import math

from tests.path_optimizer import PathOptimizer


class PIDController:
    """ 单层 PID 控制器 """
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


class PIDPathFollower:
    """
    使用 PID 控制路径跟踪，优化终止条件，避免提前降落
    """
    def __init__(self, client, max_deviation=0.5):
        self.client = client
        self.path_optimizer = PathOptimizer(max_deviation=max_deviation)  # 路径优化器

        # PID 控制器（仅单层 PID）
        self.pid_x = PIDController(1.0, 0.02, 0.3, 2.0, 2.5)
        self.pid_y = PIDController(1.0, 0.02, 0.3, 2.0, 2.5)
        self.pid_z = PIDController(1.2, 0.03, 0.3, 2.0, 3.0)
        self.pid_yaw = PIDController(0.6, 0.01, 0.1, 1.0, 1.5)

    def get_current_state(self):
        """ 获取无人机当前状态（位置、速度、偏航角） """
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation

        # 计算偏航角
        _, _, yaw = self.quaternion_to_eulerian_angles(orientation)
        return pos, vel, yaw

    @staticmethod
    def quaternion_to_eulerian_angles(q):
        """ 四元数 -> 偏航角 """
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0, 0, yaw  # 仅关心偏航角

    @staticmethod
    def compute_yaw_angle(current_yaw, target_yaw):
        """
        计算从 current_yaw 旋转到 target_yaw 的最短角度
        """
        delta_yaw = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi  # 计算最短旋转角
        return delta_yaw  # 返回旋转角度（范围 -π ~ π）

    def rotate_to_target_pid(self, target_yaw, dt=0.05, yaw_tol=math.radians(3)):
        """
        使用 PID 控制器旋转无人机到目标朝向
        """
        print(f"旋转到目标角度: {math.degrees(target_yaw):.2f}°")

        self.pid_yaw.reset()
        stable_counter = 0  # 旋转稳定计数
        required_stable = 5  # 需要连续 5 次稳定才认为旋转完成

        while True:
            _, _, current_yaw = self.get_current_state()
            yaw_error = self.compute_yaw_angle(current_yaw, target_yaw)

            # 计算 PID 控制的旋转速度
            yaw_rate = self.pid_yaw.compute(yaw_error, dt)

            # 发送旋转指令
            self.client.moveByVelocityAsync(
                0, 0, 0, dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))
            )

            # 判断是否到达目标角度
            if abs(yaw_error) < yaw_tol:
                stable_counter += 1
            else:
                stable_counter = 0

            # 确保旋转稳定
            if stable_counter >= required_stable:
                print(f"偏航调整完成，目标角度: {math.degrees(target_yaw):.2f}°")
                break

            time.sleep(dt)

    def move_along_path(self, path, dt=0.05, pos_tol=0.1, yaw_tol=math.radians(5), vel_tol=0.05, use_optimization=True):
        """
        使用 PID 控制沿路径飞行，优化终止条件
        :param path: 原始路径点
        :param dt: 控制时间步长
        :param pos_tol: 位置误差容忍度
        :param yaw_tol: 偏航误差容忍度
        :param vel_tol: 速度误差容忍度
        :param use_optimization: 是否优化路径（默认开启）
        """
        global target_yaw
        if not path or len(path) < 2:
            print("路径为空或点数过少，无法飞行")
            return

        if use_optimization:
            print("优化路径中...")
            path = self.path_optimizer.optimize_path(path)
            print(f"优化完成，路径点数: {len(path)}")

        for i in range(len(path)):
            target_x, target_y, target_z = path[i]
            print(f"目标点 {i + 1}/{len(path)} -> ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
            self.move_to_target(target_x, target_y, target_z, 0, dt, pos_tol, yaw_tol, vel_tol)

            # 到达后计算下一个航点的偏航角
            if i < len(path) - 1:
                next_x, next_y, _ = path[i + 1]
                target_yaw = math.atan2(next_y - target_y, next_x - target_x)
                self.rotate_to_target_pid(target_yaw)  # 使用 PID 控制旋转

    def move_to_target(self, target_x, target_y, target_z, target_yaw, dt, pos_tol, yaw_tol, vel_tol):
        """
        PID 控制无人机移动到目标点，优化终止条件
        """
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

        stable_counter = 0  # 目标点稳定计数
        required_stable = 5  # 需要连续 5 次稳定才认为到达

        while True:
            pos, vel, current_yaw = self.get_current_state()

            # 计算误差
            error_x = target_x - pos.x_val
            error_y = target_y - pos.y_val
            error_z = target_z - pos.z_val
            yaw_error = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi

            # 计算 PID 速度指令
            vx = self.pid_x.compute(error_x, dt)
            vy = self.pid_y.compute(error_y, dt)
            vz = self.pid_z.compute(error_z, dt)
            yaw_rate = self.pid_yaw.compute(yaw_error, dt)

            # 发送非阻塞运动指令
            self.client.moveByVelocityAsync(
                vx, vy, vz, dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))
            )

            # 目标点稳定性检查
            pos_distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)
            vel_magnitude = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)

            if pos_distance < pos_tol and abs(yaw_error) < yaw_tol and vel_magnitude < vel_tol:
                stable_counter += 1
            else:
                stable_counter = 0

            # 确保真正稳定
            if stable_counter >= required_stable:
                print(f"到达目标点 ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
                break

            time.sleep(dt)
