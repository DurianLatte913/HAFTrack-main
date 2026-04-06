import torch
import torch.nn as nn
import torch.nn.functional as F
class DownsampleBlock(nn.Module):
    """下采样模块：通过卷积层降低分辨率（步长=2），增加通道数"""
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)  # 新增 Dropout2d

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class UpsampleBlock(nn.Module):
    """上采样模块：通过反卷积或插值恢复分辨率，调整通道数"""
    def __init__(self, in_channels, out_channels, upsample_mode="deconv", kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.dropout = nn.Dropout2d(p=0.2)  # 新增 Dropout2d
        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # 激活函数（根据需求选择）
        # 反卷积上采样（带参数学习）
        if upsample_mode == "deconv":
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1
            )
            
        # 双线性/最近邻插值（无参数，轻量）"linear" or "nearest"
        else:      
            self.upsample = nn.Upsample(scale_factor=stride, mode=upsample_mode, align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 调整通道数

    def forward(self, x):
        if self.upsample_mode == "deconv":
            x = self.upsample(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.dropout(x) 
        else:
            x = self.upsample(x)
            x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x