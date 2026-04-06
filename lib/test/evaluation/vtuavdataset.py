import numpy as np
from lib.test.evaluation.data import Sequence, Sequence_RGBT, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VTUAVDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vtuav_path
        self.st_path = os.path.join(self.base_path, "test")
        # self.st_path = "/home/zhaojiacong/datasets/ST_val_split.txt"
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        anno_path = sequence_info['anno_path']
        ground_truth_rect = load_text(str(anno_path), delimiter=[' ', '\t', ','], dtype=np.float64)
        img_list_v = sorted([p for p in os.listdir(os.path.join(sequence_path, 'rgb')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_v = [os.path.join(sequence_path, 'rgb', img) for img in img_list_v]
        
        img_list_i = sorted([p for p in os.listdir(os.path.join(sequence_path, 'ir')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])
        frames_i = [os.path.join(sequence_path, 'ir', img) for img in img_list_i]
        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence_RGBT(sequence_info['name'], frames_v, frames_i, 'vtuav', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)
        
    def _get_sequence_list(self):
        # 测试集根目录（self.st_path应指向测试集的根文件夹）
        root_dir = self.st_path
    
        # 动态扫描根目录下的所有子目录，筛选有效序列
        sequence_list = []
        if os.path.isdir(root_dir):  # 确保根目录存在
            # 遍历根目录下的所有条目
            for entry in os.scandir(root_dir):
                # 只处理子目录（每个序列对应一个子目录）
                if entry.is_dir(follow_symlinks=False):
                    sequence_name = entry.name
                    # 校验该序列目录下是否存在必要的标注文件（根据原代码，需要rgb.txt）
                    anno_path = os.path.join(entry.path, 'rgb.txt')
                    if os.path.exists(anno_path):
                        # 若存在标注文件，则视为有效序列
                        sequence_list.append(sequence_name)
    
        # 按名称排序（可选，保证顺序一致性）
        sequence_list.sort()
    
        # 生成序列信息列表（保持原有格式）
        sequence_info_list = []
        for seq_name in sequence_list:
            sequence_info = {
                "name": seq_name,
                "path": os.path.join(root_dir, seq_name),
                "anno_path": os.path.join(root_dir, seq_name, 'rgb.txt')
            }
            sequence_info_list.append(sequence_info)
    
        return sequence_info_list