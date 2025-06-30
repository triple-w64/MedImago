# PyTorch 超分辨率模型集成说明

## 概述

已成功为 `sr.py` 中的SuperResolutionTab添加了IPGSPAN和SPAN PyTorch模型支持。

## 新增功能

### 支持的PyTorch模型

1. **SPAN** - EchoSPANNet模型
   - 模型文件: `Model/echospannet_scale2_best.pth`
   - 网络结构: `Model/echospan_net.py` 中的 `EchoSPANNet`

2. **IPGSPAN** - IPGGNNEchoSPANNet模型  
   - 模型文件: `Model/ipggnnechospannet_scale2_best.pth`
   - 网络结构: `Model/FPRAEcho.py` 中的 `IPGGNNEchoSPANNet`

### 主要修改

#### 1. 导入和依赖
```python
import torch
import torch.nn as nn
from PIL import Image

# 动态导入网络结构
from echospan_net import EchoSPANNet
from FPRAEcho import IPGGNNEchoSPANNet
```

#### 2. 模型加载系统重构
- 分离了OpenCV DNN模型和PyTorch模型的加载逻辑
- 支持两种模型类型的自动检测和切换
- 添加了模型清理机制防止内存泄漏

#### 3. 推理系统
- OpenCV模型: 继续使用原有的3通道BGR推理流程
- PyTorch模型: 使用单通道灰度图推理流程
- 自动处理图像预处理和后处理

#### 4. 错误处理和状态显示
- 详细的模型加载状态信息
- 兼容性检查和错误提示
- GPU/CPU自动选择

## 使用方法

1. 在超分辨率标签页中选择算法: `SPAN` 或 `IPGSPAN`
2. 选择放大倍数: 2x (目前支持)
3. 加载图像或捕获视频帧
4. 点击"应用超分辨率"进行推理

## 技术特性

### SPAN (EchoSPANNet)
- 基于SPAtion-aware blocks的超分辨率网络
- 支持傅立叶域处理
- 针对超声图像优化

### IPGSPAN (IPGGNNEchoSPANNet)  
- 集成了图神经网络(GNN)的增强版本
- 支持智能patch处理
- 更高的重建质量

## 系统要求

- PyTorch >= 1.8.0
- CUDA支持(可选，CPU也可运行)
- 对应的预训练模型文件

## 故障排除

1. **模型导入失败**: 检查Model目录下是否有正确的网络结构文件
2. **模型文件不存在**: 确保 `.pth` 文件在正确位置
3. **GPU内存不足**: 系统会自动切换到CPU模式
4. **推理失败**: 检查输入图像格式和尺寸

## 性能优化

- 自动GPU内存管理
- 模型缓存避免重复加载
- 高效的图像预处理流程

---

*集成完成时间: 2024年12月*
*支持的PyTorch版本: 2.6.0+* 