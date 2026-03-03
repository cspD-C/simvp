"""
model.py - SimVP 模型架构定义

这个文件定义了 SimVP (Simpler yet Better Video Prediction) 的完整模型架构:

核心组件:
1. Encoder: 编码器 - 将输入视频帧编码为潜在表示
2. Decoder: 解码器 - 将潜在表示解码为预测的未来帧
3. Mid_Xnet: 中间网络 - 在潜在空间进行时间建模
4. SimVP: 完整模型 - 组合以上三个组件

模型流程:
输入视频序列 -> Encoder -> Mid_Xnet -> Decoder -> 预测未来帧

论文: SimVP: Simpler yet Better Video Prediction (CVPR 2022)
"""

import torch
from torch import nn
from modules import ConvSC, Inception

# ============================================================================
# 辅助函数: stride_generator - 生成步长序列
# ============================================================================
# 作用: 为编码器和解码器生成下采样/上采样的步长序列
# 参数:
#   N: 需要的步长数量
#   reverse: 是否反转序列(解码器使用)
# 返回: [1,2,1,2,...] 的步长列表
# 示例: N=4 -> [1,2,1,2] (编码器) 或 [2,1,2,1] (解码器)
# ============================================================================

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]


# ============================================================================
# 组件1: Encoder - 编码器
# ============================================================================
# 作用: 将输入的视频帧编码为低维潜在表示
# 功能:
#   - 通过多层卷积逐步下采样
#   - 提取空间特征
#   - 保存第一层特征用于跳跃连接(skip connection)
# 输入: [B*T, C, H, W] - 批次中的所有帧
# 输出: 
#   - latent: [B*T, C_hid, H', W'] - 编码后的潜在表示
#   - enc1: [B*T, C_hid, H', W'] - 第一层特征(用于解码器)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1



# ============================================================================
# 组件2: Decoder - 解码器
# ============================================================================
# 作用: 将潜在表示解码为预测的视频帧
# 功能:
#   - 通过转置卷积逐步上采样
#   - 使用跳跃连接融合编码器特征
#   - 恢复原始图像分辨率
# 输入:
#   - hid: [B*T, C_hid, H', W'] - 来自 Mid_Xnet 的潜在表示
#   - enc1: [B*T, C_hid, H', W'] - 来自编码器的跳跃连接特征
# 输出: [B*T, C, H, W] - 预测的视频帧
# ============================================================================

class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


# ============================================================================
# 组件3: Mid_Xnet - 中间网络(时间建模核心)
# ============================================================================
# 作用: 在潜在空间进行时间序列建模,这是 SimVP 的核心创新
# 架构: U-Net 风格的编码器-解码器结构
# 功能:
#   - 使用 Inception 模块进行多尺度时间特征提取
#   - 编码器: 逐步压缩时间信息
#   - 解码器: 逐步恢复时间信息
#   - 跳跃连接: 保留不同层次的时间特征
# 输入: [B, T, C, H, W] - 时间序列的潜在表示
# 输出: [B, T, C, H, W] - 建模后的潜在表示
# 关键: 这个模块负责学习视频中的时间动态和运动模式
# ============================================================================

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y



# ============================================================================
# 完整模型: SimVP - Simpler yet Better Video Prediction
# ============================================================================
# 作用: 组合 Encoder、Mid_Xnet、Decoder 构成完整的视频预测模型
# 
# 模型流程:
#   1. Encoder: 将输入帧编码为潜在表示
#   2. Mid_Xnet: 在潜在空间进行时间建模
#   3. Decoder: 将建模后的潜在表示解码为预测帧
#
# 输入: [B, T, C, H, W] - 输入视频序列
#   B: 批次大小
#   T: 时间步数(输入帧数)
#   C: 通道数
#   H, W: 图像高度和宽度
#
# 输出: [B, T, C, H, W] - 预测的未来视频序列
#
# 特点:
#   - 端到端训练,使用 MSE 损失
#   - 纯 CNN 架构,无 RNN/Transformer
#   - 简单但有效
# ============================================================================

class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y