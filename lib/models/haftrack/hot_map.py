import torch
import cv2
import numpy as np
import os

def generate_heatmap_from_features(feats, target_size=(256, 256), save_dir='./heatmaps', filename='heatmap.jpg'):
    """
    输入: feats: shape (Batch, 256, 768) 或 (256, 768) 的真实特征 tensor
    输出: 将生成的热力图保存到指定目录
    """
    # 0. 确保保存文件夹存在
    os.makedirs(save_dir, exist_ok=True)

    if feats.dim() == 3:
        feats = feats.squeeze(0) 

    seq_len, channels = feats.shape
    spatial_side = int(np.sqrt(seq_len)) 

    feats_2d = feats.view(spatial_side, spatial_side, channels)
    heatmap_gray = torch.max(feats_2d, dim=2)[0] 

    heatmap_gray = heatmap_gray - heatmap_gray.min()
    heatmap_gray = heatmap_gray / (heatmap_gray.max() + 1e-8) 

    heatmap_np = heatmap_gray.cpu().detach().numpy()

    heatmap_resized = cv2.resize(heatmap_np, target_size, interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    save_path = os.path.join(save_dir, filename)
    abs_path = os.path.abspath(save_path)
    
    success = cv2.imwrite(abs_path, heatmap_colored) 
    
    return heatmap_colored

    fname = f"frame_{random.randint(1000, 9999)}.jpg"
    generate_heatmap_from_features(
            feats=x_hcm.detach(), # 传入网络真实的特征
            target_size=(256, 256),
            save_dir='./vis_results',
            filename=fname
    )