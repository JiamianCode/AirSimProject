import csv

class LabelManager:
    def __init__(self, csv_file):
        self.id_to_label = {}  # 用于存储物体 ID 到标签的映射
        self.label_to_ids = {}  # 用于存储标签到物体 ID 的映射
        self.load_labels(csv_file)

    def load_labels(self, csv_file):
        """
        从 CSV 文件加载标签和 ID 映射
        """
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) != 2:
                    continue  # 跳过不符合格式的行
                object_id, label = row
                # 处理 ID 为数组格式，去掉空格并按逗号分隔
                object_id = list(map(int, object_id.strip('[]').split()))  # 修正为列表
                if label not in self.label_to_ids:
                    self.label_to_ids[label] = []
                # 确保一个 id 只对应一个 label
                if tuple(object_id) not in self.id_to_label:
                    self.id_to_label[tuple(object_id)] = label
                    self.label_to_ids[label].append(object_id)
                else:
                    if self.id_to_label[tuple(object_id)] != label:
                        print(f"Warning: Inconsistent label for ID {object_id}, skipping.")
                    # 如果 ID 对应了不同的标签，则跳过

    def get_label_by_id(self, object_id):
        """
        根据物体 ID 查询标签
        """
        return self.id_to_label.get(tuple(object_id), None)     # 查不到返回None，也可以是别的类型例如str
