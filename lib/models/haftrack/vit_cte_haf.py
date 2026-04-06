"""
两模态的注意力权重相互指导，相互增强
"""

import math
import logging
from functools import partial
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens
from .vit import VisionTransformer
from lib.models.haftrack.block import CTEBlock, HAF_Block
from lib.models.haftrack.utils import recover_tokens
from lib.utils.misc import NestedTensor

from .position_encoding import build_position_encoding
from .jsloss import RGBTAttentionAlignment,get_remained_indices
from .hot_map import generate_heatmap_from_features
_logger = logging.getLogger(__name__)


class VisionTransformer_CTE_HAF(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', cte_loc=None, js_loc=None, cte_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.L_t = 192

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.cte_loc = cte_loc
        self.js_loc = js_loc
        feat_size=256
        mark = False
        for i in range(depth):
            cte_keep_ratio_i = 1.0
            if mark:
                try:
                    feat_size = int(math.ceil(feat_size*cte_keep_ratio[ce_index]))
                except:
                    feat_size = int(math.ceil(feat_size*cte_keep_ratio[-1]))
                mark=False
            if cte_loc is not None and i in cte_loc:
                cte_keep_ratio_i = cte_keep_ratio[ce_index]
                mark=True
                ce_index += 1
            
            if i>=11:
                blocks.append(
                    HAF_Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=cte_keep_ratio_i, layer_num=i, feat_size=feat_size+192,
                        )
                    )
            else:
                blocks.append(
                    CTEBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=cte_keep_ratio_i, layer_num=i
                        )
                    )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        # 不能学习的位置编码
        self.match_dim = 64
        self.pos_emb = build_position_encoding(self.match_dim)

        self.init_weights(weight_init)

    def forward_features(self, z_rgb, z_tir, x_rgb, x_tir, mask_z=None, mask_x=None,
                         cte_template_mask=None, cte_keep_rate=None,
                         return_last_attn=False, weight=0.5
                         ):
        template_num = 3
        global_index_s_layers = []

        B, H, W = x_rgb.shape[0], x_rgb.shape[2], x_rgb.shape[3]

        x_rgb = self.patch_embed(x_rgb)
        x_tir = self.patch_embed(x_tir)
        patch_z_rgb = []
        patch_z_tir = []
        for i in range(0, template_num): 
            patch_z_rgb.append(self.patch_embed(z_rgb[i]))
            patch_z_tir.append(self.patch_embed(z_tir[i]))
        z_rgb = patch_z_rgb
        z_tir = patch_z_tir

        N = math.ceil(x_rgb.shape[1]**0.5)      # 向上取整
        mask = torch.zeros([B,N,N], dtype=torch.bool).cuda()
        pos_emb = self.pos_emb(NestedTensor(x_rgb, mask))   # B,C,H,W
        pos_emb = pos_emb.flatten(2).transpose(-1,-2)  # BxHWxC
        
        N = math.ceil(z_rgb[0].shape[1]**0.5)
        mask = torch.zeros([B,N,N], dtype=torch.bool).cuda()
        pos_emb_z = self.pos_emb(NestedTensor(z_rgb[0], mask))   # B,C,H,W
        pos_emb_z = pos_emb_z.flatten(2).transpose(-1,-2)  # BxHWxC

        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)
        
        for i in range(0, template_num):
            z_rgb[i] += self.pos_embed_z
            z_tir[i] += self.pos_embed_z
        x_rgb += self.pos_embed_x
        x_tir += self.pos_embed_x

        x_rgb = torch.cat([z_rgb[0] ,z_rgb[1], z_rgb[2], x_rgb], dim=1)
        x_tir = torch.cat([z_tir[0] ,z_tir[1], z_tir[2], x_tir], dim=1)

        x_rgb = self.pos_drop(x_rgb)
        x_tir = self.pos_drop(x_tir)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z*template_num - 1, lens_z*template_num).to(x_rgb.device).to(torch.int64)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x_rgb.device).to(torch.int64)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        pos_emb_input = pos_emb.clone()
        hcm_rgb = x_tir.clone()
        attn_rgb_li = []; attn_tir_li = []
        js_loss = torch.tensor(0.0, device=x_rgb.device, dtype=x_rgb.dtype, requires_grad=True)
        JS_list = torch.linspace(0.2, 1.2, 12).cuda()
        Alignment = RGBTAttentionAlignment()
        for i, blk in enumerate(self.blocks):
            x_rgb, hcm_rgb, global_index_t, global_index_s, removed_index_s, attn_rgb, attn_tir, js_loss = \
                blk(x_rgb, hcm_rgb, global_index_t, global_index_s, \
                        mask_x, cte_template_mask, cte_keep_rate, js_loss)
            
            if self.js_loc is not None and i in self.js_loc:
                B, hn, H, W = attn_rgb.shape  # [32, 12, 448, 448]
                removed_index_all = torch.cat(removed_indexes_s, dim=1)
                indices = get_remained_indices(removed_index_all)
                indices = indices.to(attn_rgb.device) 
                L_t = self.L_t                 
                corrmap_rgb_st = attn_rgb[:, :, :L_t, L_t:]  # [B, 12, 192, 256] - template-search区域
                corrmap_tir_st = attn_tir[:, :, :L_t, L_t:]  # [B, 12, 192, 256] - template-search区域
                js_loss = js_loss + Alignment(corrmap_rgb_st, corrmap_tir_st, indices, indices)*JS_list[i]

            if self.cte_loc is not None and i in self.cte_loc:
                pos_emb_input = pos_emb.gather(1, global_index_s.unsqueeze(-1).repeat(1,1,self.match_dim)).clone()
                removed_indexes_s.append(removed_index_s)
                if not self.training:
                    global_index_s_layers.append(global_index_s)

        x_rgb = self.norm(x_rgb)
        hcm_rgb = self.norm(hcm_rgb)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        z_rgb = x_rgb[:, :lens_z_new]
        x_rgb = x_rgb[:, lens_z_new:]
        z_hcm = hcm_rgb[:, :lens_z_new]
        x_hcm = hcm_rgb[:, lens_z_new:]
        
        # 将shape还原
        if removed_indexes_s and removed_indexes_s[0] is not None:
            x_rgb = recovertoken(global_index_s, removed_indexes_s, lens_x, lens_x_new, lens_z_new, B, self.cat_mode, x_rgb)
        
        # re-concatenate with the template, which may be further used by other modules
        
        x_rgb = torch.cat([z_rgb, x_rgb], dim=1)

        aux_dict_rgb = {
            "attn": attn_rgb_li,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s_layers,
            "js_loc": self.js_loc,
        }

        # 将shape还原
        if removed_indexes_s and removed_indexes_s[0] is not None:
            x_hcm = recovertoken(global_index_s, removed_indexes_s, lens_x, lens_x_new, lens_z_new, B, self.cat_mode,x_hcm)
 
        # re-concatenate with the template, which may be further used by other modules
        x_hcm = torch.cat([z_hcm, x_hcm], dim=1)

        aux_dict_tir = {
            "attn": attn_tir_li,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s_layers,
            "js_loc": self.js_loc,
        }

        return x_rgb, x_hcm, aux_dict_rgb, aux_dict_tir, js_loss


    def forward(self, z_rgb,z_tir, x_rgb,x_tir, cte_template_mask=None, cte_keep_rate=None, return_last_attn=False, weight=0.5):

        return self.forward_features(z_rgb=z_rgb, z_tir=z_tir,
                                     x_rgb=x_rgb, x_tir=x_tir,
                                     cte_template_mask=cte_template_mask, cte_keep_rate=cte_keep_rate, weight=weight)




def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformer_CTE_HAF(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            except:
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                except:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_cte_haf(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_cte(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

def recovertoken(global_index_s, removed_indexes_s, lens_x, lens_x_new, lens_z_new, B, cat_mode,x):
    removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

    pruned_lens_x = lens_x - lens_x_new
    pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
    x = torch.cat([x, pad_x], dim=1)
    index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
    # recover original token order
    C = x.shape[-1]
    # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
    x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

    x = recover_tokens(x, lens_z_new, lens_x, mode=cat_mode)
    return x