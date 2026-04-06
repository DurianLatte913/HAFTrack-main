import torch
import torch.nn as nn
from .tan import TReFPN

def build_fpn(cfg):
    if cfg.MODEL.FPN.TYPE == 'TREFPN':
        return TReFPN(in_dim=cfg.MODEL.HIDDEN_DIM,
                    num_blocks=cfg.MODEL.FPN.NBLOCKS,
                    pre_conv=cfg.MODEL.FPN.PRE_CONV,
                    num_heads=cfg.MODEL.FPN.NHEADS,
                    mlp_ratio=cfg.MODEL.FPN.MLP_RATIOS, 
                    qkv_bias=cfg.MODEL.FPN.QKV_BIAS,
                    act_layer=nn.GELU, 
                    norm_layer=nn.LayerNorm, 
                    )
    else:
        print('FPN not implemented')