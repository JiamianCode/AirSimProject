import torch
from torch.nn.functional import conv2d
import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from scipy.ndimage import label as cpu_label


def preprocess_depth(depth_array, neighborhood_size=3):
    """
    深度图预处理：空洞填充和中值滤波，使用 GPU 加速。
    Args:
        depth_array (np.ndarray): 深度图，二维数组。
        neighborhood_size (int): 中值滤波的窗口大小。

    Returns:
        torch.Tensor: 预处理后的深度图（在 GPU 上）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_tensor = torch.tensor(depth_array, device=device, dtype=torch.float32)

    # 空洞填充
    mask = (depth_tensor > 0).float()  # 掩码：深度值为 0 的地方
    kernel = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    depth_sum = conv2d(depth_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    valid_count = conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    depth_filled = depth_tensor.clone()
    depth_filled[valid_count.squeeze() > 0] = (depth_sum / valid_count).squeeze()[valid_count.squeeze() > 0]

    # 替代 median_pool2d 的中值滤波实现
    pad = neighborhood_size // 2
    padded_depth = torch.nn.functional.pad(depth_filled, (pad, pad, pad, pad), mode="constant", value=0)
    unfolded_depth = padded_depth.unfold(0, neighborhood_size, 1).unfold(1, neighborhood_size, 1)
    depth_filtered = torch.median(unfolded_depth.contiguous().view(-1, neighborhood_size ** 2), dim=1)[0]
    depth_filtered = depth_filtered.view(depth_filled.shape)

    return depth_filtered


def extract_valid_points(depth_tensor, depth_threshold=20.0):
    """
    使用 GPU 提取有效点
    Args:
        depth_tensor (torch.Tensor): 深度图（在 GPU 上）。
        depth_threshold (float): 深度值阈值。

    Returns:
        torch.Tensor: 有效点坐标和深度值的张量 (N, 3)。
    """
    indices = torch.nonzero((depth_tensor > 0) & (depth_tensor < depth_threshold), as_tuple=False)
    depths = depth_tensor[indices[:, 0], indices[:, 1]].unsqueeze(1)
    valid_points = torch.cat((indices.float(), depths), dim=1)  # (i, j, depth)
    return valid_points


def fit_plane_ransac(points):
    """
    使用 RANSAC 在 GPU 上拟合平面。
    Args:
        points (torch.Tensor): 有效点坐标和深度值的张量 (N, 3)。

    Returns:
        tuple: 平面法向量、平面偏移量、内点掩码。
    """
    if points.shape[0] < 3:
        raise ValueError("Valid points are too few to fit a plane")

    # 转移到 CPU，使用 sklearn 的 RANSAC
    points_cpu = points.cpu().numpy()
    X = points_cpu[:, :2]  # 前两列为像素坐标 (i, j)
    y = points_cpu[:, 2]  # 第三列为深度值 (z)
    ransac = RANSACRegressor(residual_threshold=1)
    ransac.fit(X, y)

    # 获取拟合结果
    plane_normal = np.append(ransac.estimator_.coef_, -1)  # 平面法向量
    plane_offset = ransac.estimator_.intercept_  # 平面偏移量
    inlier_mask = ransac.inlier_mask_  # 内点掩码
    return plane_normal, plane_offset, inlier_mask


# GPU 加速的剔除小连通区域函数
def remove_small_connected_components_gpu(mask, min_area):
    """
    使用 GPU 剔除掩码中的小连通区域。

    Args:
        mask: torch.Tensor
            二值掩码，形状为 [H, W]，在 GPU 上。
        min_area: int
            最小区域阈值，仅保留面积大于等于该值的连通区域。

    Returns:
        torch.Tensor: 剔除小区域后的掩码，形状为 [H, W]。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将掩码移到 CPU 上，使用 SciPy 的连通区域标记函数
    mask_cpu = mask.cpu().numpy()
    labeled_mask, num_labels = cpu_label(mask_cpu)

    # 统计每个连通区域的面积
    area_sizes = [np.sum(labeled_mask == label_id) for label_id in range(1, num_labels + 1)]

    # 创建新的掩码，仅保留面积大于等于 min_area 的区域
    cleaned_mask_cpu = np.zeros_like(mask_cpu, dtype=np.uint8)
    for label_id, area_size in enumerate(area_sizes, start=1):
        if area_size >= min_area:
            cleaned_mask_cpu[labeled_mask == label_id] = 1  # 保留大区域

    # 将处理后的掩码移回 GPU
    cleaned_mask = torch.tensor(cleaned_mask_cpu, device=device, dtype=torch.uint8)

    return cleaned_mask


# 可视化单个平面
def visualize_single_plane(plane_mask, category_name, plane_id):
    # 将掩码移到 CPU 并可视化
    plane_mask_cpu = plane_mask.cpu().numpy()
    plt.imshow(plane_mask_cpu, cmap='gray')
    plt.title(f"Plane {plane_id}: {category_name}")
    plt.axis("off")
    plt.show()


def step2(depth, allowed_masks, unknown_masks, depth_threshold=20.0, min_points_per_plane=5000):
    """
        对深度图进行平面检测，使用 GPU 加速部分过程。
        Args:
            depth (np.ndarray): 深度图。
            allowed_masks (dict): 允许类别的掩码字典。
            unknown_masks (dict): 未知类别的掩码字典。
            depth_threshold (float): 深度阈值。
            min_points_per_plane (int): 每个平面的最小点数。

        Returns:
            tuple: 允许平面和未知平面列表。
        """

    def process_masks(mask_dict, preprocessed_depth):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        planes = []
        plane_id = 0  # 平面编号
        for category, mask in mask_dict.items():
            # 剔除小连通区域
            mask = remove_small_connected_components_gpu(mask, min_points_per_plane)

            # 转为 GPU 张量并提取深度
            masked_depth = torch.where(mask.bool(), preprocessed_depth, torch.tensor(0.0, device=device))

            # 提取有效点
            valid_points = extract_valid_points(masked_depth, depth_threshold)

            # 如果没有有效点，跳过当前掩码
            if valid_points.shape[0] == 0:
                continue

            # 使用 RANSAC 拟合平面
            try:
                plane_normal, plane_offset, inlier_mask = fit_plane_ransac(valid_points)
                inlier_points = valid_points[inlier_mask]

                # 如果内点不足，跳过当前平面
                if inlier_points.shape[0] < min_points_per_plane:
                    continue

                # 打印平面信息
                # print(f"  - Category: {category}")
                # print(f"  - Normal Vector: {plane_normal}")
                # print(f"  - Offset: {plane_offset}")
                # print(f"  - Inlier Points: {inlier_points.shape[0]}")

                # 可视化平面
                # visualize_single_plane(mask, category, plane_id)

                # 保存平面信息
                planes.append({
                    "mask": mask,  # 原始掩码
                    "normal": plane_normal,
                    "offset": plane_offset,
                    "category": category
                })
                plane_id += 1

            except ValueError:
                continue
        return planes

    # 预处理深度图
    preprocessed_depth = preprocess_depth(depth)

    # 处理允许掩码和平面掩码
    allowed_planes = process_masks(allowed_masks, preprocessed_depth)
    unknown_planes = process_masks(unknown_masks, preprocessed_depth)

    return allowed_planes, unknown_planes
