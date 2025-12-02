#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

# 预定义Sobel算子（全局缓存，避免重复创建）
_sobel_x = None
_sobel_y = None

def gradient_loss(img1, img2, normalize=True):
    """
    计算图像梯度损失，鼓励边缘和纹理更锐利
    使用Sobel算子检测边缘，对梯度图计算L1损失
    
    Args:
        img1, img2: 输入图像对
        normalize: 是否对梯度图进行归一化，使其与像素值量级一致
    
    性能优化：使用缓存的Sobel算子，减少重复创建开销
    """
    global _sobel_x, _sobel_y
    
    def get_gradient(img):
        # 确保输入是4D (B, C, H, W) 或 3D (C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        # 使用缓存的Sobel算子
        global _sobel_x, _sobel_y
        if _sobel_x is None or _sobel_x.device != img.device or _sobel_x.dtype != img.dtype:
            # 只在第一次或设备/类型变化时创建
            _sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
            _sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        
        # 对每个通道计算梯度
        channels = img.shape[-3]
        grad_x = F.conv2d(img, _sobel_x.repeat(channels, 1, 1, 1), 
                         padding=1, groups=channels)
        grad_y = F.conv2d(img, _sobel_y.repeat(channels, 1, 1, 1), 
                         padding=1, groups=channels)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # 归一化：将梯度图缩放到与像素值相同的量级 [0, 1]
        # Sobel算子在纯黑白边界上的最大响应约为4.0（对于[0,1]范围的图像）
        if normalize:
            grad_mag = grad_mag / 4.0
            grad_mag = torch.clamp(grad_mag, 0, 1)
        
        return grad_mag
    
    grad1 = get_gradient(img1)
    grad2 = get_gradient(img2)
    return l1_loss(grad1, grad2)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_raw(img1, img2, window_size=11):
    """
    Compute per-pixel SSIM dissimilarity map (1 - ssim_map).
    Returns a [B, 1, H, W] tensor for element-wise weighting.
    Used by background-weighted loss to emphasize peripheral regions.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return dissimilarity map (1 - ssim), averaged over channels → [B, 1, H, W]
    dssim_map = 1.0 - ssim_map.mean(dim=1, keepdim=True)
    return dssim_map

