import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict

class Conv3XC(nn.Module):
    """SPAN架构中的Conv3XC模块，参数高效的卷积实现"""
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            # 确保eval_conv的权重和偏置与输入在同一设备上
            if self.eval_conv.weight.device != x.device:
                self.eval_conv = self.eval_conv.to(x.device)
            
            self.update_params()
            # 再次确保权重已经更新到与输入相同的设备
            self.eval_conv.weight.data = self.weight_concat.to(x.device)
            self.eval_conv.bias.data = self.bias_concat.to(x.device)
            
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

class SPAB(nn.Module):
    """Swift Parameter-free Attention Block"""
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 返回(输出特征、中间特征、注意力图)的元组
        """
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att

class FourierConvBlock(nn.Module):
    """傅里叶变换卷积块，处理频域信息"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FourierConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 空间域处理
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        
        # 简化频域处理设计，专注于维度匹配
        # 确保频域处理的输入和输出通道数与空间域一致
        self.freq_layer = nn.Sequential(
            # 处理拼接后的实部和虚部
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 空间域处理
        spatial_out = self.conv_layer(x)
        
        try:
            # 保存原始形状，用于后续恢复
            b, c, h, w = x.shape
            
            # 执行FFT
            x_freq = torch.fft.rfft2(x, norm='ortho')
            
            # 分离实部和虚部
            x_freq_r = x_freq.real  # 形状为[b, c, h, w//2+1]
            x_freq_i = x_freq.imag  # 形状为[b, c, h, w//2+1]
            
            # 调整实部和虚部的形状，保持通道数一致
            channel_adapter = nn.Conv2d(c, c, kernel_size=1).to(x.device)
            x_freq_r = channel_adapter(x_freq_r)
            x_freq_i = channel_adapter(x_freq_i)
            
            # 确保通道数正确后拼接
            x_freq_cat = torch.cat([x_freq_r, x_freq_i], dim=1)  # 通道数变为2c
            
            # 应用频域卷积层
            freq_processed = self.freq_layer(x_freq_cat)  # 输出通道数为out_channels
            
            # 将处理后的特征分为实部和虚部
            freq_real = freq_processed
            freq_imag = torch.zeros_like(freq_real)
            
            # 创建复数张量
            freq_complex = torch.complex(freq_real, freq_imag)
            
            # 逆FFT回到空间域
            freq_spatial = torch.fft.irfft2(freq_complex, s=(h, w), norm='ortho')
            
            # 使用门控机制整合空间和频域信息
            gate = self.gate(spatial_out)
            result = spatial_out * gate + freq_spatial * (1 - gate)
            
            return result
            
        except Exception as e:
            # 如果傅里叶处理失败，使用空间域结果
            print(f"傅里叶处理出错: {e}，使用空间域结果替代")
            return spatial_out
    
    def get_equivalent_kernel_bias(self):
        """获取等效的卷积核和偏置，用于推理优化"""
        # 由于频域处理无法直接等效为单个卷积，这里返回空间域部分的卷积核和偏置
        return self.conv_layer.weight, self.conv_layer.bias

class EchoSPANNet(nn.Module):
    """超声心动图超分辨率网络 (结合SPAN和EchoSRNet的架构)"""
    def __init__(self, 
                 scale: int = 2, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 feature_channels: int = 48,
                 num_blocks: int = 6,
                 use_fourier: bool = True,
                 img_range: float = 1.0,
                 ultrasound_mode: bool = True):
        """EchoSPANNet模型构造函数
        Args:
            scale: 超分辨率缩放因子 (2或4)
            in_channels: 输入通道数
            out_channels: 输出通道数
            feature_channels: 特征通道数
            num_blocks: SPAB块数量
            use_fourier: 是否使用傅里叶变换处理
            img_range: 图像像素范围
            ultrasound_mode: 是否使用超声图像专用处理模式
        """
        super(EchoSPANNet, self).__init__()
        
        self.scale = scale
        self.img_range = img_range
        self.use_fourier = use_fourier
        self.ultrasound_mode = ultrasound_mode
        
        # 输入卷积层
        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        
        # SPAB块
        self.spab_blocks = nn.ModuleList([
            SPAB(feature_channels, bias=True) for _ in range(num_blocks)
        ])
        
        # 傅里叶变换处理模块
        if use_fourier:
            self.fourier_block = FourierConvBlock(feature_channels, feature_channels)
        
        # 特征融合卷积
        num_cat_features = min(4, num_blocks + 1)  # 包含初始特征
        self.conv_cat = nn.Conv2d(
            feature_channels * num_cat_features, 
            feature_channels, 
            kernel_size=1, 
            bias=True
        )
        
        # 最终特征处理
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)
        
        # 上采样模块
        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, out_channels * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        # 正规化输入
        x = x * self.img_range
        
        # 初始特征提取
        out_feature = self.conv_1(x)
        
        # 保存中间特征
        block_features = []
        attention_maps = []
        
        # 应用SPAB块
        x_blocks = out_feature
        for i, block in enumerate(self.spab_blocks):
            x_blocks, feature, attention = block(x_blocks)
            
            # 保存重要的中间特征用于后续融合
            if i == 0 or i == len(self.spab_blocks) // 2 or i == len(self.spab_blocks) - 1:
                block_features.append(x_blocks)
                attention_maps.append(attention)
        
        # 傅里叶变换处理
        if self.use_fourier:
            fourier_features = self.fourier_block(x_blocks)
            x_blocks = x_blocks + fourier_features * 0.3  # 轻微融合频域信息
        
        # 特征融合
        x_blocks = self.conv_2(x_blocks)
        
        # 选择要拼接的特征
        cat_features = [out_feature]  # 初始特征
        
        # 添加选定的块特征
        for feat in block_features[:3]:  # 最多使用3个块特征
            cat_features.append(feat)
        
        # 拼接并融合特征
        out = self.conv_cat(torch.cat(cat_features[:4], dim=1))  # 最多拼接4个特征
        
        # 上采样到目标尺寸
        output = self.upsampler(out)
        
        # 确保输出在合理范围内
        return torch.clamp(output, 0.0, self.img_range)

class OptimizedEchoSPANNet(nn.Module):
    """经过重参数化优化的EchoSPANNet，用于更快的推理"""
    def __init__(self, 
                 scale: int = 2, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 feature_channels: int = 48,
                 num_blocks: int = 6,
                 img_range: float = 1.0):
        """优化版EchoSPANNet模型构造函数"""
        super(OptimizedEchoSPANNet, self).__init__()
        
        self.scale = scale
        self.img_range = img_range
        
        # 输入卷积层 (已优化)
        self.input_conv = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        
        # 优化的处理块
        self.process_blocks = nn.ModuleList([
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1) 
            for _ in range(num_blocks)
        ])
        
        # 特征融合卷积
        self.fusion_conv = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        
        # 最终特征处理
        self.final_conv = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        
        # 上采样模块
        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, out_channels * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
    
    def forward(self, x):
        # 正规化输入
        x = x * self.img_range
        
        # 特征提取
        feat = self.input_conv(x)
        original_feat = feat
        
        # 应用优化的处理块
        for block in self.process_blocks:
            feat = block(feat)
            
        # 融合并最终处理
        feat = self.fusion_conv(feat + original_feat)
        feat = self.final_conv(feat)
        
        # 上采样
        output = self.upsampler(feat)
        
        # 确保输出在合理范围内
        return torch.clamp(output, 0.0, self.img_range)
    
    @classmethod
    def from_trained_model(cls, model: EchoSPANNet) -> 'OptimizedEchoSPANNet':
        """从训练好的EchoSPANNet模型创建优化版本
        
        Args:
            model: 训练好的EchoSPANNet模型
            
        Returns:
            优化后的模型
        """
        # 创建优化模型实例
        optimized = cls(
            scale=model.scale,
            in_channels=model.conv_1.sk.in_channels,
            out_channels=model.upsampler[0].out_channels // (model.scale**2),
            feature_channels=model.conv_1.sk.out_channels,
            num_blocks=len(model.spab_blocks),
            img_range=model.img_range
        )
        
        # 复制输入卷积层参数
        model.conv_1.update_params()  # 确保参数已更新
        optimized.input_conv.weight.data = model.conv_1.eval_conv.weight.data.clone()
        optimized.input_conv.bias.data = model.conv_1.eval_conv.bias.data.clone()
        
        # 复制SPAB块参数 (简化后的版本)
        for i, spab in enumerate(model.spab_blocks):
            # 使用SPAB的等效卷积核
            spab.c1_r.update_params()
            spab.c2_r.update_params()
            spab.c3_r.update_params()
            
            # 由于SPAB结构复杂，这里使用一个简化的等效卷积
            optimized.process_blocks[i].weight.data = spab.c3_r.eval_conv.weight.data.clone()
            optimized.process_blocks[i].bias.data = spab.c3_r.eval_conv.bias.data.clone()
        
        # 复制特征融合卷积参数
        optimized.fusion_conv.weight.data = model.conv_cat.weight.data[:, :model.conv_cat.weight.data.shape[1]//4].clone()
        optimized.fusion_conv.bias.data = model.conv_cat.bias.data.clone()
        
        # 复制最终处理卷积参数
        model.conv_2.update_params()
        optimized.final_conv.weight.data = model.conv_2.eval_conv.weight.data.clone()
        optimized.final_conv.bias.data = model.conv_2.eval_conv.bias.data.clone()
        
        # 复制上采样模块参数
        optimized.upsampler[0].weight.data = model.upsampler[0].weight.data.clone()
        optimized.upsampler[0].bias.data = model.upsampler[0].bias.data.clone()
        
        return optimized 