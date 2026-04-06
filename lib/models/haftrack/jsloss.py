import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBTAttentionAlignment(nn.Module):
    def __init__(self, grid_size=16, sigma=1.0, temperature=1.0, eps=1e-8):
        super(RGBTAttentionAlignment, self).__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.temperature = temperature
        self.eps = eps
        self.register_buffer('spatial_prior', self._generate_gaussian_2d())

    def _generate_gaussian_2d(self):
        coords = torch.arange(self.grid_size).float()
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        center = (self.grid_size - 1) / 2
        mask = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * self.sigma**2))
        return mask.view(-1)

    def normalize_for_js(self, attn_slice):
        if self.temperature != 1.0:
            attn_slice = torch.pow(attn_slice + self.eps, 1.0 / self.temperature)

        sum_val = attn_slice.sum(dim=-1, keepdim=True) + self.eps
        p_norm = attn_slice / sum_val
        
        return torch.clamp(p_norm, min=self.eps)

    def forward(self, p_raw, q_raw, p_indices, q_indices):
        device = p_raw.device

        p_spatial = p_raw.mean(dim=(1)).sum(dim=1)
        q_spatial = q_raw.mean(dim=(1)).sum(dim=1)

        # 获取对应的空间先验权重
        p_indices = p_indices.to(device)
        q_indices = q_indices.to(device)
        prior = self.spatial_prior.to(device)
        weight_p = prior[p_indices]
        weight_q = prior[q_indices]
        
        #归一化为概率分布
        p = self.normalize_for_js(p_spatial)
        q = self.normalize_for_js(q_spatial)
        
        #计算 JS 散度
        m = (p + q) / 2.0
        m = torch.clamp(m, min=self.eps)
        
        kl_p = F.kl_div(m.log(), p, reduction='none')  # KL(p || m)
        kl_q = F.kl_div(m.log(), q, reduction='none')  # KL(q || m)
        
        js_per_token = 0.5 * (kl_p + kl_q) # [B, N_s]
        
        # 只有靠近中心的 Token 对齐误差才会被放大
        combined_weight = (weight_p + weight_q) / 2.0
        
        loss = (js_per_token * combined_weight).sum(dim=-1) / (combined_weight.sum(dim=-1) + self.eps)
        
        return loss

def get_remained_indices(removed_indices, total_tokens=256):
    """
    removed_indices: [B, N_removed] 被剔除的索引
    """
    device = removed_indices.device
    batch_size = removed_indices.shape[0]
    mask = torch.ones(batch_size, total_tokens, device=device, dtype=torch.bool)
    mask.scatter_(1, removed_indices, False)
    remained_indices = torch.nonzero(mask, as_tuple=False)[:, 1].view(batch_size, -1)
    
    return remained_indices