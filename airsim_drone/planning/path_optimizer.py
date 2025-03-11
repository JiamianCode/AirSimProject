import numpy as np
from scipy.interpolate import splprep, splev


class PathOptimizer:
    """
    处理 A* 生成的离散路径，使其更加平滑和优化
    """
    def __init__(self, max_deviation=0.3):
        """
        :param max_deviation: 最大允许的路径偏差（米）
        """
        self.max_deviation = max_deviation

    def simplify_path(self, path):
        """
        使用 Douglas-Peucker 算法减少路径点数量
        :param path: [(x, y, z), (x2, y2, z2), ...]
        :return: 简化后的路径
        """
        path = np.array(path)

        def rdp(points, epsilon):
            """
            递归简化路径点
            """
            if len(points) < 3:
                return points

            start, end = points[0], points[-1]
            line_vec = end - start
            line_length = np.linalg.norm(line_vec)

            # 保护性检查，避免除零错误
            if line_length == 0:
                return np.array([start, end])

            line_vec /= line_length  # 归一化

            # 计算每个点到直线的垂直距离
            distances = np.linalg.norm(np.cross(points - start, line_vec), axis=1)

            # 确保索引不越界
            if len(distances) == 0:
                return np.array([start, end])

            max_idx = np.argmax(distances)
            max_dist = distances[max_idx]

            if max_dist > epsilon and max_idx < len(points):  # 修正 max_idx 越界错误
                left = rdp(points[:max_idx + 1], epsilon)
                right = rdp(points[max_idx:], epsilon)
                return np.vstack((left[:-1], right))
            else:
                return np.array([start, end])

        return rdp(path, self.max_deviation)

    @staticmethod
    def smooth_path(path, smoothing_factor=0.5):
        """
        使用 B 样条曲线平滑路径
        :param smoothing_factor:
        :param path: [(x, y, z), (x2, y2, z2), ...]
        :return: 平滑后的路径
        """
        path = np.array(path)

        # 如果路径点小于 4 个，不进行平滑，直接返回
        if len(path) < 4:
            return path

        try:
            tck, u = splprep([path[:, 0], path[:, 1], path[:, 2]], s=smoothing_factor)
            u_fine = np.linspace(0, 1, len(path) * 5)
            smooth_path = splev(u_fine, tck)
            return np.vstack(smooth_path).T
        except Exception as e:
            print(f"平滑路径失败: {e}")
            return path  # 如果平滑失败，则返回原始路径

    def optimize_path(self, path):
        """
        综合优化路径，先简化后平滑
        """
        if not path or len(path) < 2:
            print("无法优化路径：路径为空或点数过少")
            return path

        simplified_path = self.simplify_path(path)
        smooth_path = self.smooth_path(simplified_path)
        return smooth_path
