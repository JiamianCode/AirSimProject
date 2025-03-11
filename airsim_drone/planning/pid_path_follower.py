import airsim
import time
import math

from airsim_drone.planning.path_optimizer import PathOptimizer
from airsim_drone.visualization.visualization import visualize_voxel_grid_with_path


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

    # 错误，可弃
    def rotate_to_target_pid_old(self, target_yaw, dt=0.05, yaw_tol=math.radians(3)):
        """
        使用 PID 控制器旋转无人机到目标朝向
        """
        print(f"旋转到目标角度: {math.degrees(target_yaw):.2f}°")

        self.pid_yaw.reset()
        stable_counter = 0  # 旋转稳定计数
        required_stable = 3  # 需要连续 5 次稳定才认为旋转完成

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
    # 可弃
    def move_along_path_old(self, voxel_manager, path, dt=0.05, pos_tol=0.1, yaw_tol=math.radians(5), vel_tol=0.05,
                        use_optimization=True, visualize=False):
        """
        使用 PID 控制沿路径飞行，优化终止条件
        :param voxel_manager: 网格管理类
        :param path: 原始路径点
        :param dt: 控制时间步长
        :param pos_tol: 位置误差容忍度
        :param yaw_tol: 偏航误差容忍度
        :param vel_tol: 速度误差容忍度
        :param use_optimization: 是否优化路径（默认开启）
        :param visualize: 是否可视化（默认关闭）
        """
        if not path or len(path) < 2:
            print("路径为空或点数过少，无法飞行")
            return

        if use_optimization:
            print("优化路径中...")
            path = self.path_optimizer.optimize_path(path)
            print(f"优化完成，路径点数: {len(path)}")

        if visualize:
            visualize_voxel_grid_with_path(voxel_manager, path)

        last_yaw = None  # 记录上一个目标的朝向

        for i in range(len(path)):
            target_x, target_y, target_z = path[i]

            # 计算朝向角度（目标偏航角）
            if i < len(path) - 1:
                next_x, next_y, _ = path[i + 1]
                target_yaw = math.atan2(next_y - target_y, next_x - target_x)  # 计算朝向
                last_yaw = target_yaw  # 记录当前的目标朝向
            else:
                target_yaw = last_yaw  # 最后一个点使用倒数第二个点的角度

            print(f"目标点 {i + 1}/{len(path)} -> ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), 目标角度: {math.degrees(target_yaw):.2f}°")

            # 先旋转到目标方向
            if target_yaw is not None:
                self.rotate_to_target_pid(target_yaw)

            # 再移动到目标点
            self.move_to_target(target_x, target_y, target_z, target_yaw, dt, pos_tol, yaw_tol, vel_tol)

    def rotate_to_target_pid(self, target_yaw, dt=0.05, yaw_tol=math.radians(3), alt_tol=0.05):
        """
        使用 PID 控制器旋转无人机到目标朝向，同时维持当前高度
        :param target_yaw: 目标偏航角 (弧度)
        :param dt: 控制时间步长
        :param yaw_tol: 允许的偏航误差 (弧度)
        :param alt_tol: 允许的高度误差 (米)
        """
        print(f"旋转到目标角度: {math.degrees(target_yaw):.2f}°")

        # 重置 PID 控制器
        self.pid_yaw.reset()
        self.pid_z.reset()

        stable_counter = 0  # 旋转稳定计数
        required_stable = 3  # 需要连续 3 次稳定才认为旋转完成

        # 读取当前高度
        pos, _, _ = self.get_current_state()
        target_z = pos.z_val  # 目标高度（保持当前高度）

        while True:
            pos, _, current_yaw = self.get_current_state()
            yaw_error = self.compute_yaw_angle(current_yaw, target_yaw)
            alt_error = target_z - pos.z_val  # 计算高度误差（正值表示无人机下降，需要向上修正）

            # 计算 PID 控制输出
            yaw_rate = self.pid_yaw.compute(yaw_error, dt)  # 偏航控制
            vz = self.pid_z.compute(alt_error, dt)  # 维持高度控制

            # 发送旋转指令，同时调整高度
            self.client.moveByVelocityAsync(
                0, 0, vz, dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))
            )

            # 判断是否稳定
            if abs(yaw_error) < yaw_tol and abs(alt_error) < alt_tol:
                stable_counter += 1
            else:
                stable_counter = 0

            # 旋转稳定
            if stable_counter >= required_stable:
                print(f"偏航调整完成，目标角度: {math.degrees(target_yaw):.2f}°，高度保持: {target_z:.2f}m")
                break

            time.sleep(dt)

    def move_along_path(self, voxel_manager, path, dt=0.05, pos_tol=0.1, yaw_tol=math.radians(5), vel_tol=0.05,
                        use_optimization=True, visualize=False, d=1.0, D=3, yaw_tolerance=math.radians(20)):
        """
        让无人机沿路径飞行，并保持朝向目标，同时检测转向情况
        :param voxel_manager: 体素网格管理器
        :param path: 规划的路径 [(x, y, z), ...]
        :param dt: 控制步长
        :param pos_tol: 位置误差允许范围
        :param yaw_tol: 偏航角误差允许范围
        :param vel_tol: 速度误差允许范围
        :param use_optimization: 是否优化路径
        :param visualize: 是否可视化
        :param d: 每次最大飞行距离
        :param D: 降落前的剩余距离
        :param yaw_tolerance: 角度误差容忍范围，超过该值会先执行转向
        :return: 是否完成降落 (True: 已降落, False: 需要继续飞行)
        """
        if not path or len(path) < 2:
            print("路径为空或点数过少，无法飞行")
            return False

        if use_optimization:
            print("优化路径中...")
            path = self.path_optimizer.optimize_path(path)
            print(f"优化完成，路径点数: {len(path)}")

        if visualize:
            visualize_voxel_grid_with_path(voxel_manager, path)

        # **计算路径总长度 S**
        total_length = 0
        segment_lengths = []
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            seg_length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2) ** 0.5
            segment_lengths.append(seg_length)
            total_length += seg_length

        print(f"路径总长度: {total_length:.2f} m")

        # **确定飞行的路径长度**
        if total_length > D:
            flight_length = min(d, total_length - D)  # 不能超过总长度 - D
            full_flight = False  # 仅飞行一段
        else:
            flight_length = total_length  # 直接飞完全程
            full_flight = True  # 可以降落

        print(f"本次飞行目标距离: {flight_length:.2f} m")

        # **路径选择逻辑修正**
        if full_flight:
            selected_path = path  # 直接使用完整路径，不进行截取
        else:
            traveled_length = 0
            selected_path = [path[0]]  # 选定的飞行路径
            for i in range(len(path) - 1):
                if traveled_length + segment_lengths[i] >= flight_length:
                    # 计算最终点的插值
                    ratio = (flight_length - traveled_length) / segment_lengths[i]
                    last_x = path[i][0] + ratio * (path[i + 1][0] - path[i][0])
                    last_y = path[i][1] + ratio * (path[i + 1][1] - path[i][1])
                    last_z = path[i][2] + ratio * (path[i + 1][2] - path[i][2])
                    selected_path.append((last_x, last_y, last_z))
                    break
                else:
                    traveled_length += segment_lengths[i]
                    selected_path.append(path[i + 1])

        print(f"飞行路径点数: {len(selected_path)}")

        # **计算当前角度**
        _, _, current_yaw = self.get_current_state()
        last_yaw = None  # 记录上一个目标的朝向

        for i in range(len(selected_path)):
            target_x, target_y, target_z = selected_path[i]

            # 计算目标角度
            if i < len(selected_path) - 1:
                next_x, next_y, _ = selected_path[i + 1]
                target_yaw = math.atan2(next_y - target_y, next_x - target_x)  # 计算朝向
                last_yaw = target_yaw  # 记录当前的目标朝向
            else:
                target_yaw = last_yaw  # 最后一个点使用倒数第二个点的角度

            yaw_error = abs(self.compute_yaw_angle(current_yaw, target_yaw))

            # **如果需要转向，先旋转并返回 False**
            if yaw_error > yaw_tolerance:
                print(f"需要转向 {math.degrees(yaw_error):.2f}° (当前: {math.degrees(current_yaw):.2f}°, 目标: {math.degrees(target_yaw):.2f}°)，执行旋转")
                self.rotate_to_target_pid(target_yaw)
                return False  # 仅完成旋转，未继续飞行

            print(f"目标点 {i + 1}/{len(selected_path)} -> ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}), 目标角度: {math.degrees(target_yaw):.2f}°")

            # 先旋转到目标方向
            if target_yaw is not None:
                self.rotate_to_target_pid(target_yaw)

            # 再移动到目标点
            self.move_to_target(target_x, target_y, target_z, target_yaw, dt, pos_tol, yaw_tol, vel_tol)

        return full_flight  # 返回是否降落完成


    def move_to_target(self, target_x, target_y, target_z, target_yaw, dt, pos_tol, yaw_tol, vel_tol):
        """
        PID 控制无人机移动到目标点，优化终止条件
        """
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

        stable_counter = 0  # 目标点稳定计数
        required_stable = 3  # 需要连续 5 次稳定才认为到达

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
