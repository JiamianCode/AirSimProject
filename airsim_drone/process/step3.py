import torch
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion, label as cpu_label


def detect_planes(points, filter_config):
    """
    使用 RANSAC 从点云中提取多个平面，并返回法向量等信息。
    """
    distance_threshold = filter_config.get('distance_threshold', 0.03)
    max_iterations = filter_config.get('max_iterations', 5000)
    min_points = filter_config.get('min_points', 50)
    max_planes = filter_config.get('max_planes', 5)

    planes = []
    points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)
    remaining_indices = np.arange(len(points_np))

    for _ in range(max_planes):
        if len(remaining_indices) < min_points:
            break

        current_points = points_np[remaining_indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(current_points)

        try:
            plane_model, local_inliers = pcd.segment_plane(distance_threshold, min_points, max_iterations)
        except RuntimeError:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal_norm = normal / np.linalg.norm(normal)
        global_inliers = remaining_indices[local_inliers]

        inlier_mask = np.zeros(len(points_np), dtype=bool)
        inlier_mask[global_inliers] = True

        if isinstance(points, torch.Tensor):
            normal = torch.from_numpy(normal).to(points.device).float()
            inlier_mask = torch.tensor(inlier_mask, device=points.device)

        planes.append({
            "normal": normal,
            "offset": d,
            "inlier_mask": inlier_mask,
            "inlier_points": points[global_inliers] if isinstance(points, torch.Tensor) else points_np[global_inliers]
        })

        remaining_indices = np.delete(remaining_indices, local_inliers)

    return planes


def step3(predictions, metadata, points, valid_indices, filter_config):
    """
    提取 "wall", "window" 等类别的掩码, 进行侵蚀处理, 在点云中提取平面，并返回法向量等信息。
    """
    target_categories = {"wall", "window "}
    sem_seg = predictions["sem_seg"].argmax(dim=0)
    device = sem_seg.device
    extracted_planes = []

    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device)
    if not isinstance(valid_indices, torch.Tensor):
        valid_indices = torch.from_numpy(valid_indices).to(device)

    for category_id, category_name in enumerate(metadata.stuff_classes):
        if category_name not in target_categories:
            continue

        mask = (sem_seg == category_id)
        if mask.sum().item() < filter_config.get("min_area", 500):
            continue

        mask_np = mask.cpu().numpy()
        kernel_size = filter_config.get("erosion_kernel_size", 3)
        iterations = filter_config.get("erosion_iterations", 2)
        for _ in range(iterations):
            mask_np = binary_erosion(mask_np, structure=np.ones((kernel_size, kernel_size)))

        mask = torch.from_numpy(mask_np).to(device)
        labeled_mask, num_labels = cpu_label(mask_np)
        labeled_mask = torch.from_numpy(labeled_mask).to(device)

        u = valid_indices[:, 0].long()
        v = valid_indices[:, 1].long()
        region_labels = labeled_mask[v, u]

        for label_id in range(1, num_labels + 1):
            mask_per_label = (region_labels == label_id)
            region_points = points[mask_per_label]

            if len(region_points) < filter_config.get("min_points", 50):
                continue

            detected_planes = detect_planes(region_points, filter_config)

            for plane_info in detected_planes:
                plane_info["category"] = category_name
                plane_info["mask"] = mask
                plane_info["valid_indices"] = valid_indices[mask_per_label]
                extracted_planes.append(plane_info)

    return extracted_planes