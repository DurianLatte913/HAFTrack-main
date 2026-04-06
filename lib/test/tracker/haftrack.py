import math

from lib.models.haftrack import build_haftrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target

import cv2
import os
import torch.nn.functional as F
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.tracker.modal_contribution import ReadModalContribution
from lib.vis.attn_map_vis import recover_and_tomap,attn_on_image


class HAFTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(HAFTrack, self).__init__(params)
        network = build_haftrack(params.cfg, training=False)
        try:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        except:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        self.cfg = params.cfg
        self.update = params.update
        self.update_param = params.update_param
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.threshold = 1
        if params.debug or params.update!=None:
            self.contribution = ReadModalContribution(self.network)

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.box_mask_z = {}
        self.template_frames = []
        self.max_template_frames = 2
        self.initial_frame = None
        self.next_template_rgb = []
        self.next_template_tir = []
        self.next_box_mask_z = []
    def initialize(self, image_v, image_i, info: dict):
        self.contrs = None
        # forward the template once
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        # z_patch_arr_tir=z_patch_arr_rgb
        # z_patch_arr_rgb=z_patch_arr_tir

        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir
        
        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        with torch.no_grad():
            self.z_dict1_rgb = template_rgb
            self.z_dict1_tir = template_tir
            self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]
        box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CTE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_rgb,
                                                        template_rgb.tensors.device).squeeze(1)
            box_mask_z = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

        self.initial_frame = [template_rgb, template_tir]
        self.template_frames = [self.initial_frame,self.initial_frame]
        self.box_mask_z = torch.cat([box_mask_z,box_mask_z,box_mask_z], dim=1)
        self.next_template_rgb = template_rgb
        self.next_template_tir = template_tir
        self.next_box_mask_z = box_mask_z
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


    def update_template(self, image_v, image_i):
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir
        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        with torch.no_grad():
            self.z_dict1_rgb = template_rgb
            self.z_dict1_tir = template_tir
            self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]
        #提取模板中心区域
        if self.cfg.MODEL.BACKBONE.CTE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(self.state, resize_factor_rgb,
                                                        template_rgb.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

    def track(self, image_v, image_i, info: dict = None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        #搜索帧
        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)
        
        #用于更新模板,每50帧更新一次模版
        if self.frame_id % 50 == 0:
            #更新模板帧
            self.template_frames[0] = self.template_frames[1]
            self.template_frames[1] = [self.next_template_rgb, self.next_template_tir]
            size = 64
            _,N = self.box_mask_z.shape
            self.box_mask_z = torch.cat([self.box_mask_z[:,0:size],self.box_mask_z[:,N-size:N],self.next_box_mask_z],dim=1)
            
            z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
            z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.next_template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
            self.next_template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
            if self.cfg.MODEL.BACKBONE.CTE_LOC:
                template_bbox_rgb = self.transform_bbox_to_crop(self.state, resize_factor_rgb,
                                                        self.next_template_rgb.tensors.device).squeeze(1)
                self.next_box_mask_z = generate_mask_cond(self.cfg, 1, self.next_template_rgb.tensors.device, template_bbox_rgb)
        
        with torch.no_grad():
            x_dict_rgb = search_rgb
            x_dict_tir = search_tir

            x_dict = [x_dict_rgb,x_dict_tir]
            # merge the template and the search
            # run the transformer
            # out_dict = self.network.forward(
            #     template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
            all_template_rgb = torch.cat([self.initial_frame[0].tensors.unsqueeze(0),self.template_frames[0][0].tensors.unsqueeze(0),self.template_frames[1][0].tensors.unsqueeze(0)],dim=0)
            all_template_tir = torch.cat([self.initial_frame[1].tensors.unsqueeze(1),self.template_frames[0][1].tensors.unsqueeze(0),self.template_frames[1][1].tensors.unsqueeze(0)],dim=0)
            all_template = torch.cat([all_template_rgb, all_template_tir],dim=0)
            out_dict = self.network.forward(
                template=all_template, 
                search=[x_dict[0].tensors, x_dict[1].tensors], 
                cte_template_mask=self.box_mask_z)
            pred_score_map = out_dict['score_map']
            response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)

        
        if self.debug:
            # self.visdom.register(torch.stack([torch.stack(cell) for cell in self.contribution[150]], dim=0), 'lineplot', 1, 'contribution')
            self.visdom.register(self.contribution[250], 'lineplot', 1, 'contribution')

        if self.update!=None:
            score = response.max()
            m = self.contribution[1][0]
            if(self.update == "timing"):
                if(self.frame_id % self.update_param['timing'] == 0):
                    self.update_template(image_v, image_i)
            elif(self.update == "thr"):
                self.threshold = self.update_param['thr'][0]
                if not self.update_param['lower']:
                    if (score > self.threshold).item():
                        self.update_template(image_v, image_i)
                else:
                    if (score < self.threshold).item():
                        self.update_template(image_v, image_i)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                # self.visdom.register((image_v, info['gt_bbox'].tolist(), self.state, info['baseline_rect'].tolist()), 'Tracking', 1, 'Tracking')
                self.visdom.register((image_v, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                search_image = torch.cat([torch.from_numpy(x_patch_arr_rgb),torch.from_numpy(x_patch_arr_tir)], dim=1).permute(2, 0, 1)
                self.visdom.register(search_image, 'image', 1, 'search_region')
                # self.visdom.register(torch.from_numpy(x_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'search_region_t')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'template_v')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'template_t')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                # 注意力权重可视化
                if 0:
                    # 数据提取
                    rgb_attn_map = []
                    tir_attn_map = []
                    for blk in self.network.backbone.blocks:
                        if hasattr(blk, 'attn_map'):
                            # blk_num += 1
                            rgb_attn_map.append(blk.attn_map['RGB'])
                            tir_attn_map.append(blk.attn_map['TIR'])
                    # 格式化
                    # 层 -> 选择attn_map -> H,W 
                    layer = -1
                    a=2
                    
                    # before
                    r_b = torch.from_numpy(attn_on_image(
                        np.array(recover_and_tomap(rgb_attn_map[layer][0], out_dict['aux_dict_rgb']['global_index_s'][a], vis='st').squeeze().unsqueeze(-1).cpu()),
                        x_patch_arr_rgb)).permute(2,0,1)
                    t_b = torch.from_numpy(attn_on_image(
                        np.array(recover_and_tomap(tir_attn_map[layer][0], out_dict['aux_dict_tir']['global_index_s'][a], vis='st').squeeze().unsqueeze(-1).cpu()),
                        x_patch_arr_tir)).permute(2,0,1)
                    before = torch.cat([r_b, t_b], dim=-1)
                    self.visdom.register(before ,'image', 1, 'attn-before')
                    # after
                    r_a = torch.from_numpy(attn_on_image(
                        np.array(recover_and_tomap(rgb_attn_map[layer][2], out_dict['aux_dict_rgb']['global_index_s'][a], vis='st').squeeze().unsqueeze(-1).cpu()),
                        x_patch_arr_rgb)).permute(2,0,1)
                    t_a = torch.from_numpy(attn_on_image(
                        np.array(recover_and_tomap(tir_attn_map[layer][2], out_dict['aux_dict_tir']['global_index_s'][a], vis='st').squeeze().unsqueeze(-1).cpu()),
                        x_patch_arr_tir)).permute(2,0,1)
                    after = torch.cat([r_a, t_a], dim=-1)
                    self.visdom.register(after ,'image', 1, 'attn-after')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                    
                if 'aux_dict_rgb' in out_dict and out_dict['aux_dict_rgb']:
                    removed_indexes_s_v = out_dict['aux_dict_rgb']['removed_indexes_s']
                    removed_indexes_s_v = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s_v]
                    masked_search_v = gen_visualization(x_patch_arr_rgb, removed_indexes_s_v)
                    self.visdom.register(torch.from_numpy(masked_search_v).permute(2, 0, 1), 'image', 1, 'masked_search_v')
                if 'aux_dict_tir' in out_dict and out_dict['aux_dict_tir']:
                    removed_indexes_s_t = out_dict['aux_dict_tir']['removed_indexes_s']
                    removed_indexes_s_t = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s_t]
                    masked_search_t = gen_visualization(x_patch_arr_tir, removed_indexes_s_t)
                    self.visdom.register(torch.from_numpy(masked_search_t).permute(2, 0, 1), 'image', 1, 'masked_search_t')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor_rgb, resize_factor_rgb)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return HAFTrack
