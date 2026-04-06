import os
import cv2 as cv
import time
import sys
import torch
import numpy as np
sys.path.append("/opt/data/VTUAV/HAFTrack-main")
sys.path.append("/opt/data/VTUAV/HAFTrack-main/lib")
from tqdm import tqdm
from lib.vis.visdom_cus import Visdom
from lib.test.evaluation import trackerlist, get_dataset
from lib.test.utils.load_text import load_text

class VisResults(object):
    def __init__(self, save_img = False):
        self.save_img = save_img
        self._init_visdom()  # 初始化Visdom
        
    # 初始化Visdom（适配代码库原生逻辑）
    def _init_visdom(self, visdom_info=None):
        visdom_info = {'port': 8098} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False

        try:
            # 初始化自定义Visdom类（和HAFTrack中tracker的初始化逻辑一致）
            self.visdom = Visdom(
                1, 
                {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                visdom_info=visdom_info, 
                env='RGBT_Tracking_Results'
            )

            # 显示操作帮助文本
            help_text = (
                '操作说明：\n'
                '1. 选中Tracking窗口后按【空格】暂停/继续\n'
                '2. 暂停时按【右箭头】单帧步进\n'
                '3. 按【n】跳过当前序列'
            )
            self.visdom.register(help_text, 'text', 1, 'Help')
            print("Visdom初始化成功！请在浏览器访问 http://localhost:8098 查看可视化结果")
        except Exception as e:
            raise Exception(f"Visdom初始化失败：{str(e)}\n请先执行命令启动Visdom：visdom -port 8098")

    # Visdom UI事件处理（暂停/步进/跳序列）
    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':  # 空格暂停/继续
                self.pause_mode = not self.pause_mode
            elif data['key'] == 'ArrowRight' and self.pause_mode:  # 右箭头单帧步进
                self.step = True
            elif data['key'] == 'n':  # n键跳序列
                self.next_seq = True

    def vis_dataset(self, dataset, tracker_configs, skip_missing_seq=False, seq_list=[]):
        # 适配跟踪器格式：兼容原trackerlist + 手动指定路径
        trackers = []
        for cfg in tracker_configs:
            if isinstance(cfg, dict):
                # 手动指定格式：包装成和原tracker对象兼容的结构
                class DummyTracker:
                    def __init__(self, name, dir):
                        self.name = name
                        self.results_dir = dir
                trackers.append(DummyTracker(cfg['display_name'], cfg['results_dir']))
            else:
                # 原trackerlist格式：直接使用
                trackers.append(cfg)
       # 遍历数据集序列
        for seq_id, seq in enumerate(tqdm(dataset)):
            seq_name = seq.name
            if seq_list and seq_name not in seq_list:
                continue
            # 加载真值框
            anno_bb = torch.tensor(seq.ground_truth_rect)
            # 加载所有跟踪器的预测框
            all_pred_boxes = []
            tracker_names = [trk.name for trk in trackers]
            for trk_id, trk in enumerate(trackers):
                results_path = f'{trk.results_dir}/{seq.name}.txt'
                if os.path.isfile(results_path):
                    pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
                    all_pred_boxes.append(pred_bb)
                else:
                    if skip_missing_seq:
                        print(f"警告：跟踪器 {trk.name} 无 {seq_name} 结果，跳过该序列")
                        break
                    raise Exception(f'Result not found: {results_path}')
            # ========== RGB-T双模态帧路径处理 ==========
            if hasattr(seq, 'frames_v') and hasattr(seq, 'frames_i'):
                # RGBT数据集：可见光(frames_v) + 热成像(frames_i)
                frame_list_v = seq.frames_v
                frame_list_i = seq.frames_i
                assert len(frame_list_v) == len(frame_list_i), f"序列 {seq_name} 可见光/热成像帧数量不匹配！"
                frame_pairs = list(zip(frame_list_v, frame_list_i))
            elif hasattr(seq, 'frames'):
                # 单模态兼容：仅可见光
                frame_pairs = [(f, None) for f in seq.frames]
            else:
                raise AttributeError(f"序列 {seq_name} 无有效帧属性（frames_v/frames_i/frames）")
            # 逐帧可视化
            for i in range(len(anno_bb)):
                # 暂停/步进逻辑
                if self.pause_mode and not self.step:
                    while self.pause_mode:
                        time.sleep(0.01)
                self.step = False
                # 跳过当前序列
                if self.next_seq:
                    self.next_seq = False
                    break
                # ========== 读取并拼接RGB-T双模态帧 ==========
                frame_v_path, frame_i_path = frame_pairs[i]
           # 读取可见光帧
                im_v = cv.imread(frame_v_path)
                if im_v is None:
                    print(f"警告：可见光帧 {frame_v_path} 读取失败，跳过")
                    continue
                im_v = cv.cvtColor(im_v, cv.COLOR_BGR2RGB)  # BGR->RGB
             # 读取并处理热成像帧
                im_combined = im_v
                if frame_i_path is not None:
                    im_i = cv.imread(frame_i_path)
                    if im_i is not None:
                        im_i = cv.cvtColor(im_i, cv.COLOR_BGR2RGB)
                        # 缩放热成像帧至和可见光帧同尺寸
                        im_i = cv.resize(im_i, (im_v.shape[1], im_v.shape[0]))
                        # 横向拼接（左：可见光，右：热成像）
                        im_combined = np.hstack((im_v, im_i))
                vis_data = [im_combined]
                # 处理 GT 框 (防止 NaN)
                gt_box = anno_bb[i].numpy()
                if np.isnan(gt_box).any():
                    gt_box = [0, 0, 0, 0] # 遇到 NaN 用 0 替代，不在画面上乱画
                vis_data.append(gt_box.tolist())
              # 处理各个 Tracker 的预测框 (防止 NaN)
                for pred_bb in all_pred_boxes:
                    p_box = pred_bb[i].numpy()
                    if np.isnan(p_box).any():
                        p_box = [0, 0, 0, 0]
                    vis_data.append(p_box.tolist())
                # ========== Visdom注册可视化内容 ==========
                tracker_caption = ", ".join([f"{name}(红/蓝/黄/紫)" for name in tracker_names])
                caption = (
                    f"序列: {seq_name} | 帧号: {i:03d}\n"

                    f"GT(绿), {tracker_caption}"
                )
                # 核心修改：明确传入 seq_name 和 frame_id，供底层保存图片时作为纯净的文件名使用
                self.visdom.register(
                    vis_data, 'Tracking', 1, 'Tracking',
                    caption=caption,
                    seq_name=seq_name, # 新增
                    frame_id=i,         # 新增
                    save_img=self.save_img  # <--- 只有这里需要加
                ) 

    # 兼容原代码的update_boxes方法（可选保留）
    def update_boxes(self, data, caption):
        self.visdom.register(data, 'Tracking', 1, 'Tracking', caption=caption)

if __name__ == '__main__':
    # 初始化可视化器
    viser = VisResults(save_img=True)

    # ========== 配置数据集和跟踪器 ==========
    # 1. 数据集名称（替换为你的RGBT数据集，如RGBT234/GTOT/LASHER/VTUAV）
    dataset_name = 'RGBT234'
    dataset = get_dataset(dataset_name)

    # 2. 跟踪器配置（支持多个跟踪器，手动指定结果路径）
    from lib.test.evaluation.environment import env_settings
    env = env_settings()  # 获取默认结果根路径

    tracker_configs = [
    ]

    # ========== 执行可视化 ==========
    # seq_list可指定特定序列，如seq_list=['basketball_player_1']
    viser.vis_dataset(dataset, tracker_configs, skip_missing_seq=True, seq_list=['flower2'])