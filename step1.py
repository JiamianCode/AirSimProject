from matplotlib import pyplot as plt


def filter_masks_by_semantic(predictions, metadata, allow_list, block_list, min_area=500):
    """
    根据语义分割结果过滤掩码，并按类别存储掩码，同时筛除小区域。

    Args:
        predictions (dict): 包含语义分割结果的字典。
        metadata (MetadataCatalog): 包含类别和语义信息的元数据。
        allow_list (set): 允许操作的类别名称集合。
        block_list (set): 不允许操作的类别名称集合。
        min_area (int): 最小区域面积，小于此面积的掩码将被过滤掉。

    Returns:
        allowed_masks (dict): 满足要求类别的掩码字典，键为类别名称，值为类别掩码。
        unknown_masks (dict): 未知类别的掩码字典，键为类别名称，值为类别掩码。
    """
    sem_seg = predictions["sem_seg"].argmax(dim=0)  # GPU 上的张量

    # print(f"类别总数: {len(metadata.stuff_classes)}")

    allowed_masks = {}
    unknown_masks = {}

    for category_id, category_name in enumerate(metadata.stuff_classes):
        # print(category_name)

        # 生成类别掩码
        mask = (sem_seg == category_id)

        # 筛除过小的区域
        if mask.sum().item() < min_area:
            continue

        # 判断类别是否在允许列表或阻止列表中
        if category_name in allow_list:
            allowed_masks[category_name] = mask.clone()
        elif category_name not in block_list:
            unknown_masks[category_name] = mask.clone()

    return allowed_masks, unknown_masks


def allowed_masks_show(allowed_masks):
    for category_name, mask in allowed_masks.items():
        plt.figure()
        plt.title(f"Allowed Category: {category_name}")
        plt.imshow(mask, cmap="gray")  # 使用灰度显示掩码
        plt.axis("off")
        plt.show()


def unknown_masks_show(unknown_masks):
    for category_name, mask in unknown_masks.items():
        plt.figure()
        plt.title(f"Unknown Category: {category_name}")
        plt.imshow(mask, cmap="gray")  # 使用灰度显示掩码
        plt.axis("off")
        plt.show()


def step1(metadata, predictions):
    # 词库配置
    # 允许无人机降落的类别
    allow_list = {
        "floor", "grass", "road, route", "sidewalk, pavement", "table", "desk"
    }
    # 不允许无人机降落的类别
    block_list = {
        "door", "window ", "tree", "ceiling", "person", "water", "wall", "computer",
        "light", "bottle", "cabinet", "poster, posting, placard, notice, bill, card",
        "bag", "box", "bulletin board", "lamp"
    }

    # 筛选掩码
    min_area = 5000  # 例如：小于 1000 像素的区域将被过滤掉
    allowed_masks, unknown_masks = filter_masks_by_semantic(predictions, metadata, allow_list, block_list, min_area)

    # 展示掩码
    # allowed_masks_show(allowed_masks)
    # unknown_masks_show(unknown_masks)

    return allowed_masks, unknown_masks
