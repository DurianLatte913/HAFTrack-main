import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_



class CossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.saliency_bias = nn.Parameter(torch.tensor(0.0))
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.saliency_proj = nn.Sequential(
            nn.Linear(dim, dim // 16, bias=False), # 先压缩通道
            nn.LayerNorm(dim // 16),               # 归一化保证训练稳定
            nn.GELU(),                                    # 相比 ReLU，GELU 更适合 Transformer 架构
            nn.Dropout(0.1),
            nn.Linear(dim // 16, 1, bias=False)    # 投影到单通道权重
        )
        nn.init.normal_(self.saliency_proj[-1].weight, std=0.001)
        self.print_iter = 0

    def calculate_entropy(self, attn_tensor):
        return -(attn_tensor * torch.log(attn_tensor + 1e-9)).sum(dim=-1).mean().item()
    
    def forward(self, attn, rgb, return_attn=False):
        entropy_before = self.calculate_entropy(attn)
        B,N,C = rgb.shape
        tau = 0.8
        attn_sh = torch.pow(attn, 1/tau)
        attn_sh = F.normalize(attn_sh, p=1, dim=-1)
        saliency_mask = torch.sigmoid(self.saliency_proj(rgb.detach())*self.gamma + self.saliency_bias)
        attn_sh_template = attn_sh[:, :, :, :192]
        attn_sh_search = attn_sh[:, :, :, 192:]
        search_mask = saliency_mask[:,192:,:].permute(0, 2, 1).unsqueeze(1)
        attn_sh_search_masked = attn_sh_search * search_mask
        refined_attn_sh = torch.cat([attn_sh_template, attn_sh_search_masked], dim=-1)
        refined_attn_sh = F.normalize(refined_attn_sh, p=1, dim=-1)
        alpha = torch.clamp(self.gate_alpha, 0.0, 0.1)
        attn = attn + alpha * (refined_attn_sh - attn)
        attn = (1 - alpha) * attn + alpha * refined_attn_sh
        entropy_after = self.calculate_entropy(attn)

        v = self.v_linear(rgb).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def get_corrmap(self, q, k, mask):
        """
        输出原始的QK权重，未softmax
        """
        # q: B, N, C
        B, N1, C = q.shape
        B, N2, C = k.shape
        q = self.q_linear(q).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,hn,N,C/hn
        k = self.k_linear(k).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
        return attn



    def get_attn_x(self, attn, v, return_attention=False, need_softmax=True):
        """
        根据attn权重，组织value
        """
        if need_softmax:        
            attn = attn.softmax(dim=-1)
            
        B,N,C = v.shape
        v = self.v_linear(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


    def general_forward(self, query, key, value, mask, return_attention=False):
        # q: B, N, C
        B, N1, C = query.shape
        B, N2, C = key.shape
        q = self.q_linear(query).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B,hn,N,C/hn
        k = self.k_linear(key).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(value).reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x
        

    def forward(self, q, k=None, v=None, mask=None, return_attention=False):
        k = q if k==None else k
        v = q if v==None else v
        return self.general_forward(q,k,v,mask, return_attention)

