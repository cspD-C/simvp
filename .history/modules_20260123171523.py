"""
modules.py - SimVP 模型的基础构建模块

这个文件包含了 SimVP 模型的所有基础卷积模块:
1. BasicConv2d: 基础卷积层(支持普通卷积和转置卷积)
2. ConvSC: 空间卷积模块(Spatial Convolution)
3. GroupConv2d: 分组卷积层
4. Inception: Inception 模块(多尺度特征提取)

这些模块是构建 Encoder、Decoder 和 Mid_Xnet 的基础组件
"""

from torch import nn



# ============================================================================
# 模块1: BasicConv2d - 基础卷积层
# ============================================================================
# 作用: 提供基础的2D卷积操作,支持普通卷积和转置卷积(用于上采样)
# 特点: 
#   - 可选的 GroupNorm 归一化
#   - 可选的 LeakyReLU 激活函数
#   - 支持转置卷积用于解码器的上采样
# ============================================================================

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y



# ============================================================================
# 模块2: ConvSC - 空间卷积模块 (Spatial Convolution)
# ============================================================================
# 作用: SimVP 的核心空间卷积单元,用于编码器和解码器
# 特点:
#   - 固定使用 3x3 卷积核
#   - 根据 stride 自动选择下采样或上采样
#   - stride=1: 保持分辨率
#   - stride=2: 下采样(编码器)或上采样(解码器)
# ============================================================================

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y



# ============================================================================
# 模块3: GroupConv2d - 分组卷积层
# ============================================================================
# 作用: 使用分组卷积减少参数量和计算量,同时保持模型表达能力
# 特点:
#   - 将输入通道分成多个组,每组独立卷积
#   - 减少参数量: 参数量 = 原始参数量 / groups
#   - 用于 Inception 模块中的多尺度特征提取
# ============================================================================

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y



# ============================================================================
# 模块4: Inception - 多尺度特征提取模块
# ============================================================================
# 作用: 同时使用多个不同尺度的卷积核提取特征,捕获不同感受野的信息
# 特点:
#   - 使用多个卷积核尺寸 (默认: 3x3, 5x5, 7x7, 11x11)
#   - 先用 1x1 卷积降维,减少计算量
#   - 将多尺度特征相加融合
#   - 用于 Mid_Xnet 的时间建模
# 应用: 这是 SimVP 能够捕获多尺度时空特征的关键模块
# ============================================================================

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y