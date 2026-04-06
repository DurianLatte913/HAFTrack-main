import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from lib.models.layers.attn_blocks import candidate_elimination
from .attention import Attention, CossAttention
# from torch.nn.functional import *
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class CTEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                layer_num=None):
        super().__init__()

        self.layer_num = layer_num
        self.area = None
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search


    def keep_token(self, x, attn, global_index_template=None, global_index_search=None, \
                   cte_template_mask=None, keep_ratio_search=None):
        
        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search
            x, global_index_search, removed_index_search = \
                candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, cte_template_mask)

        return x, global_index_template, global_index_search, removed_index_search


    def forward(self, x_rgb, x_tir, global_index_t, global_index_s, mask, 
                cte_template_mask, keep_ratio_search, js_loss, *args, **args_dict):
        
        x_rgb_attn, corrmap_rgb = self.attn(self.norm1(x_rgb), mask=mask, return_attention=True)
        x_tir_attn, corrmap_tir = self.attn(self.norm1(x_tir), mask=mask, return_attention=True)

        x_rgb = x_rgb + self.drop_path(x_rgb_attn)
        x_tir = x_tir + self.drop_path(x_tir_attn)
        attn_all = corrmap_rgb+corrmap_tir
        x_rgb, _, _, _ = \
            self.keep_token(x_rgb, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)
        x_tir, global_index_t, global_index_s, removed_index_s = \
            self.keep_token(x_tir, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)

        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm2(x_rgb)))
        x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))

        return x_rgb, x_tir, global_index_t, global_index_s, removed_index_s, corrmap_rgb, corrmap_tir, js_loss

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                layer_num=None):
        super().__init__()

        self.layer_num = layer_num
        self.area = None
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x_rgb, x_tir, mask, pos_emb, pos_emb_z):
        
        x_rgb_attn, corrmap_rgb = self.attn(self.norm1(x_rgb), mask=mask, return_attention=True)
        x_tir_attn, corrmap_tir = self.attn(self.norm1(x_tir), mask=mask, return_attention=True)

        x_rgb = x_rgb + self.drop_path(x_rgb_attn)
        x_tir = x_tir + self.drop_path(x_tir_attn)

        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm2(x_rgb)))
        x_tir = x_tir + self.drop_path(self.mlp(self.norm2(x_tir)))

        return x_rgb, x_tir, corrmap_rgb, corrmap_tir


class HAF_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                layer_num=None,  match_dim=64, feat_size=256):
        super().__init__()
        self.layer_num = layer_num
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.inner_attn = CossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        
        self.attn_map = {'RGB':[0,0,0], 'TIR':[0,0,0]}
    


    def keep_token(self, x, attn, global_index_template=None, global_index_search=None, \
                   cte_template_mask=None, keep_ratio_search=None):
        
        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = \
                candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, cte_template_mask)

        return x, global_index_template, global_index_search, removed_index_search
    
    def forward(self, x_rgb, hcm_rgb, global_index_t, global_index_s, mask, 
                    cte_template_mask, keep_ratio_search, js_loss):
        x_rgb_attn, corrmap_rgb = self.attn(self.norm1(x_rgb), mask=mask, return_attention=True)
        x_rgb_origin = x_rgb + self.drop_path(x_rgb_attn)
        x_rgb_origin = x_rgb_origin + self.drop_path(self.mlp(self.norm2(x_rgb_origin)))
        x_aux_attn, corrmap_aux = self.attn(self.norm1(hcm_rgb), mask=mask, return_attention=True)
        hcm_rgb_attn = self.inner_attn(corrmap_aux, self.norm1(x_rgb))
        hcm_rgb_all = x_rgb + self.drop_path(hcm_rgb_attn)
        if not hasattr(self, 'norm3_inited'):
            self.norm3.load_state_dict(self.norm2.state_dict())
            self.norm3_inited = True
        hcm_rgb_all = hcm_rgb_all + self.drop_path(self.mlp(self.norm3(hcm_rgb_all)))
        x_aux = hcm_rgb + self.drop_path(x_aux_attn)
        x_aux = x_aux + self.drop_path(self.mlp(self.norm2(x_aux)))
        attn_all = corrmap_rgb + corrmap_aux
        x_rgb, _, _, _ = \
            self.keep_token(x_rgb, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)
        x_aux, global_index_t, global_index_s, removed_index_s = \
            self.keep_token(x_aux, attn_all, global_index_t, global_index_s, cte_template_mask, keep_ratio_search)
        
        return x_rgb_origin, x_aux, global_index_t, global_index_s, removed_index_s, corrmap_rgb, corrmap_aux, js_loss              