import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import time
import logging
from pathlib import Path
import cv2
import onnxruntime as ort
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from threading import Thread

# 自动设置Python路径，支持直接运行和模块导入两种方式
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 定义所需函数
def rgb_to_ycbcr(image):
    """将RGB图像转换为YCbCr色彩空间"""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return torch.stack([y, cb, cr], dim=1)

def calculate_psnr(img1, img2, max_val=1.0):
    """计算峰值信噪比"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """计算结构相似度"""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    # 获取图像的通道数
    batch_size, channels, height, width = img1.size()
    
    # 创建高斯核
    window = torch.ones(window_size, window_size) / (window_size * window_size)
    window = window.to(img1.device).view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    
    # 计算均值
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    # 计算均值的平方
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差
    sigma1_sq = torch.nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 返回平均值
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def process_image(x):
    """处理图像，用于保存和显示
    Args:
        x: 输入张量或数组
    Returns:
        处理后的图像（通常是PIL图像）
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    # 确保像素值在0-255范围内
    if x.max() <= 1.0:
        x = x * 255.0
    
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    # 处理不同的通道数
    if len(x.shape) == 2:  # 单通道灰度图
        return Image.fromarray(x, mode='L')
    elif len(x.shape) == 3 and x.shape[0] == 1:  # CHW格式的单通道
        return Image.fromarray(x[0], mode='L')
    elif len(x.shape) == 3 and x.shape[0] == 3:  # CHW格式的RGB
        return Image.fromarray(np.transpose(x, (1, 2, 0)), mode='RGB')
    elif len(x.shape) == 3 and x.shape[2] == 3:  # HWC格式的RGB
        return Image.fromarray(x, mode='RGB')
    elif len(x.shape) == 3 and (x.shape[2] == 1 or x.shape[0] == 1):  # HWC或CHW格式的单通道
        if x.shape[2] == 1:
            return Image.fromarray(x[:,:,0], mode='L')
        else:
            return Image.fromarray(x[0], mode='L')
    else:
        raise ValueError(f"不支持的图像格式: shape={x.shape}")

def save_image(img, path, quality=95):
    """保存图像到指定路径
    Args:
        img: 输入图像（PIL图像或张量）
        path: 保存路径
        quality: JPEG质量，仅对JPEG格式有效
    """
    # 如果是张量，转换为PIL图像
    if isinstance(img, torch.Tensor):
        img = process_image(img)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # 根据文件扩展名保存
    ext = os.path.splitext(path)[1].lower()
    if ext == '.jpg' or ext == '.jpeg':
        img.save(path, quality=quality)
    else:
        img.save(path)
    
    return path

# 尝试导入模型
try:
    # 当作为模块导入时使用相对导入
    from .models import EchoSRNet, SRCNN, FSRCNN, LapSRN, CARN, VDSR, EDSR, EchoSPANNet
    from .models.FPRAEcho import IPGGNNEchoSPANNet
except ImportError:
    # 当相对导入失败时，使用直接导入
    from models import EchoSRNet, SRCNN, FSRCNN, LapSRN, CARN, VDSR, EDSR, EchoSPANNet
    from models.FPRAEcho import IPGGNNEchoSPANNet


def list_pretrained_models():
    """列出trained文件夹中的所有预训练模型"""
    trained_dir = os.path.join(current_dir, 'trained')
    models = []
    
    if os.path.exists(trained_dir) and os.path.isdir(trained_dir):
        for file in os.listdir(trained_dir):
            if file.endswith('.pth'):
                models.append(file)
    
    return models

def parse_args():
    parser = argparse.ArgumentParser(description='EchoSRNet图像超分辨率推理')
    
    # 基本参数
    parser.add_argument('--input', type=str, default='/home/jm/SR/data/DIV2K/data/DIV2K_test_LR_bicubic/X2/0901x2.png', help='输入图像/视频路径，支持单图像、目录或视频文件')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    
    # 模型选择
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument('--model', type=str, help='训练好的PyTorch模型路径')
    model_group.add_argument('--onnx_model', type=str, help='ONNX模型路径')
    model_group.add_argument('--use_pretrained', type=str, help='使用trained文件夹中的预训练模型,例如: echosrnet_scale2_us_best.pth')
    model_group.add_argument('--list_pretrained', action='store_true', help='列出可用的预训练模型并退出')
    
    # GUI参数
    parser.add_argument('--gui', action='store_true', help='启动图形用户界面')
    
    # 模型参数
    parser.add_argument('--scale', type=int, default=2, help='超分辨率缩放因子 (2或4)')
    parser.add_argument('--ultrasound_mode', action='store_true', help='启用超声图像增强')
    parser.add_argument('--rgb_mode', action='store_true', help='启用RGB图像模式（处理彩色图像）')
    
    # 只有PyTorch模型需要的参数
    parser.add_argument('--m', type=int, default=5, help='线性块数量 (仅EchoSRNet模型)')
    parser.add_argument('--int_features', type=int, default=16, help='内部特征通道数量 (仅EchoSRNet模型)')
    parser.add_argument('--feature_size', type=int, default=256, help='线性块内部特征大小 (仅EchoSRNet模型)')
    
    # EchoSPANNet特有参数
    parser.add_argument('--span_feature_channels', type=int, default=48, help='EchoSPANNet: 特征通道数')
    parser.add_argument('--span_num_blocks', type=int, default=6, help='EchoSPANNet: SPAB块数量')
    parser.add_argument('--use_fourier', action='store_true', help='EchoSPANNet: 是否使用傅里叶变换处理')
    
    # IPGGNNEchoSPANNet特有参数
    parser.add_argument('--use_ipg', action='store_true', help='IPGGNNEchoSPANNet: 是否启用IPG功能')
    parser.add_argument('--max_degree', type=int, default=8, help='IPGGNNEchoSPANNet: IPG图卷积最大连接度数')
    parser.add_argument('--base_patch_size', type=int, default=64, help='IPGGNNEchoSPANNet: 基础patch大小')
    
    # 性能参数
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU推理')
    parser.add_argument('--batch_size', type=int, default=1, help='批量处理大小 (仅图像目录)')
    parser.add_argument('--num_threads', type=int, default=0, help='ONNX Runtime线程数 (0表示默认值)')
    
    # 视频处理参数
    parser.add_argument('--video', action='store_true', help='输入是视频文件')
    parser.add_argument('--video_fps', type=float, default=0, help='输出视频帧率 (0表示使用原视频帧率)')
    parser.add_argument('--video_codec', type=str, default='mp4v', help='输出视频编解码器')
    parser.add_argument('--video_bitrate', type=str, default='8000k', help='输出视频比特率')
    parser.add_argument('--skip_frames', type=int, default=0, help='处理时每N帧跳过一帧 (用于加速)')
    
    # 评估参数
    parser.add_argument('--reference', type=str, default='', help='参考HR图像/视频路径，用于计算PSNR和SSIM')
    
    # 显示参数
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--no_save', action='store_true', help='不保存结果，仅用于性能测试')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    # 测试速度参数
    parser.add_argument('--test_speed', action='store_true', help='测试模型推理速度')
    parser.add_argument('--test_iterations', type=int, default=100, help='速度测试迭代次数')
    
    return parser.parse_args()

def setup_logger(verbose=False):
    """设置日志记录器"""
    level = logging.INFO  # 始终使用INFO级别
    
    # 配置日志记录器
    logger = logging.getLogger('infer')
    logger.setLevel(level)
    
    # 清除现有处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    
    return logger

def load_pytorch_model(args, device, logger):
    """加载训练好的PyTorch模型"""
    model_path = args.model
    model_name = os.path.basename(model_path) if model_path else ""
    
    # 设置默认参数，如果不存在则使用默认值
    if not hasattr(args, 'span_feature_channels'):
        args.span_feature_channels = 48
    if not hasattr(args, 'span_num_blocks'):
        args.span_num_blocks = 6
    if not hasattr(args, 'use_fourier'):
        args.use_fourier = True
    if not hasattr(args, 'm'):
        args.m = 5
    if not hasattr(args, 'int_features'):
        args.int_features = 16
    if not hasattr(args, 'feature_size'):
        args.feature_size = 256
    if not hasattr(args, 'ultrasound_mode'):
        args.ultrasound_mode = False
    if not hasattr(args, 'use_speckle_filter'):
        args.use_speckle_filter = None
    if not hasattr(args, 'use_edge_enhancer'):
        args.use_edge_enhancer = None
    if not hasattr(args, 'use_signal_enhancer'):
        args.use_signal_enhancer = None
    # IPGGNNEchoSPANNet默认参数
    if not hasattr(args, 'use_ipg'):
        args.use_ipg = True
    if not hasattr(args, 'max_degree'):
        args.max_degree = 8
    if not hasattr(args, 'base_patch_size'):
        args.base_patch_size = 64
    
    # 确定模型类型和参数
    if 'ipggnnechospannet' in model_name.lower() or 'ipg' in model_name.lower():
        logger.info(f"加载IPGGNNEchoSPANNet模型: {model_path}")
        model = IPGGNNEchoSPANNet(
            scale=args.scale,
            in_channels=1,  # 使用Y通道训练
            out_channels=1, # 输出Y通道
            feature_channels=args.span_feature_channels,
            num_blocks=args.span_num_blocks,
            use_fourier=args.use_fourier,
            img_range=1.0,
            ultrasound_mode=args.ultrasound_mode,
            base_patch_size=args.base_patch_size,
            use_ipg=args.use_ipg,
            max_degree=args.max_degree
        )
    elif 'echospannet' in model_name.lower():
        logger.info(f"加载EchoSPANNet模型: {model_path}")
        model = EchoSPANNet(
            scale=args.scale,
            in_channels=1,  # 使用Y通道训练
            out_channels=1, # 输出Y通道
            feature_channels=args.span_feature_channels,
            num_blocks=args.span_num_blocks,
            use_fourier=args.use_fourier,
            img_range=1.0,
            ultrasound_mode=args.ultrasound_mode
        )
    elif 'fsrcnn' in model_name.lower():
        logger.info(f"加载FSRCNN模型: {model_path}")
        model = FSRCNN(
            scale=args.scale,
            num_channels=1
        )
    elif 'srcnn' in model_name.lower():
        logger.info(f"加载SRCNN模型: {model_path}")
        model = SRCNN(
            scale=args.scale,
            num_channels=1
        )
    elif 'echosrnet' in model_name.lower():
        logger.info(f"加载EchoSRNet模型: {model_path}")
        model = EchoSRNet(
            scale=args.scale,
            m=args.m,
            int_features=args.int_features,
            feature_size=args.feature_size,
            ultrasound_mode=args.ultrasound_mode,
            use_speckle_filter=args.use_speckle_filter,
            use_edge_enhancer=args.use_edge_enhancer,
            use_signal_enhancer=args.use_signal_enhancer
        )
    elif 'lapsrn' in model_name.lower():
        logger.info(f"加载LapSRN模型: {model_path}")
        model = LapSRN(
            scale=args.scale,
            num_channels=1
        )
    elif 'edsr' in model_name.lower():
        logger.info(f"加载EDSR模型: {model_path}")
        model = EDSR(
            scale=args.scale,
            num_channels=1
        )
    elif 'vdsr' in model_name.lower():
        logger.info(f"加载VDSR模型: {model_path}")
        model = VDSR(
            scale=args.scale,
            num_channels=1
        )
    else:
        logger.info(f"未识别模型类型，尝试加载为EchoSRNet: {model_path}")
        model = EchoSRNet(
            scale=args.scale,
            m=args.m,
            int_features=args.int_features,
            feature_size=args.feature_size,
            ultrasound_mode=args.ultrasound_mode,
            use_speckle_filter=args.use_speckle_filter,
            use_edge_enhancer=args.use_edge_enhancer,
            use_signal_enhancer=args.use_signal_enhancer
        )
    
    # 加载模型参数
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # 对于IPGGNNEchoSPANNet模型，可能包含动态创建的层，使用strict=False
        if 'ipggnnechospannet' in model_name.lower() or 'ipg' in model_name.lower():
            # 尝试过滤掉可能不匹配的键
            model_keys = set(model.state_dict().keys())
            state_dict_keys = set(state_dict.keys())
            
            # 找出不匹配的键
            unexpected_keys = state_dict_keys - model_keys
            missing_keys = model_keys - state_dict_keys
            
            if unexpected_keys:
                logger.warning(f"发现意外的键，将被忽略: {unexpected_keys}")
                # 移除意外的键
                for key in unexpected_keys:
                    del state_dict[key]
            
            if missing_keys:
                logger.warning(f"缺少的键，将使用默认初始化: {missing_keys}")
            
            # 使用strict=False加载
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info(f"未加载的键: {missing}")
            if unexpected:
                logger.info(f"意外的键: {unexpected}")
        else:
            # 对于其他模型，使用严格匹配
            model.load_state_dict(state_dict)
            
        logger.info("PyTorch模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
    
    # 将模型移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model

def load_onnx_model(args, logger):
    """加载ONNX模型"""
    logger.info(f"加载ONNX模型: {args.onnx_model}")
    
    try:
        # 创建ONNX运行时会话选项
        sess_options = ort.SessionOptions()
        
        # 设置线程数
        if args.num_threads > 0:
            sess_options.intra_op_num_threads = args.num_threads
            logger.info(f"ONNX Runtime线程数设置为: {args.num_threads}")
        
        # 根据CPU/GPU设置执行模式和优化级别
        if args.cpu:
            # CPU优化：关闭线程自旋等待，减少CPU使用率
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            # 设置图优化级别为最高
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            logger.info("已启用CPU优化模式，禁用线程自旋等待，启用全部图优化")
            providers = ['CPUExecutionProvider']
        else:
            # GPU优化
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("已启用GPU优化模式")
        
        # 创建ONNX运行时会话
        ort_session = ort.InferenceSession(
            args.onnx_model, 
            sess_options=sess_options,
            providers=providers
        )
        
        # 获取模型元数据
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name
        
        logger.info(f"ONNX模型输入: {input_name}, 形状: {input_shape}")
        logger.info(f"ONNX模型输出: {output_name}")
        logger.info("ONNX模型加载成功")
        
        return {
            'session': ort_session,
            'input_name': input_name,
            'output_name': output_name,
            'input_shape': input_shape
        }
    except Exception as e:
        logger.error(f"ONNX模型加载失败: {e}")
        raise

def preprocess_image(img, args, logger):
    """图像预处理，返回适合模型输入的张量"""
    img_width, img_height = img.size
    logger.info(f"原始图像尺寸: {img_width}x{img_height}")
    
    # 预处理
    if args.ultrasound_mode:
        # 超声模式：转换为YCbCr，仅保留Y通道
        input_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
        ycbcr_tensor = rgb_to_ycbcr(input_tensor)
        input_tensor = ycbcr_tensor[:, 0:1, :, :]  # 只取Y通道 [1, 1, H, W]
        if args.verbose:
            logger.info("使用超声模式 (YCbCr Y通道)")
    elif args.rgb_mode:
        # RGB模式：处理彩色图像
        input_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
        if args.verbose:
            logger.info("使用RGB模式 (处理彩色图像)")
    else:
        # 灰度模式：将图像转换为灰度
        grayscale_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        input_tensor = grayscale_transform(img).unsqueeze(0)  # [1, 1, H, W]
        if args.verbose:
            logger.info("使用灰度模式 (转换为单通道图像)")
    
    logger.info(f"预处理后张量形状: {input_tensor.shape}")
    return input_tensor

def process_with_pytorch(input_tensor, model, device, args, logger):
    """使用PyTorch模型处理图像张量"""
    # 移动到设备
    input_tensor = input_tensor.to(device)
    
    # 处理RGB模式的特殊情况
    if args.rgb_mode:
        # 分离RGB通道分别处理
        channels = []
        start_time = time.time()
        for c in range(3):
            channel_input = input_tensor[:, c:c+1, :, :]
            with torch.no_grad():
                channel_output = model(channel_input)
            channels.append(channel_output)
        
        # 合并处理后的通道
        output_tensor = torch.cat(channels, dim=1)
        inference_time = time.time() - start_time
    else:
        # 常规推理（单通道）
        start_time = time.time()
        with torch.no_grad():
            output_tensor = model(input_tensor)
        inference_time = time.time() - start_time
    
    logger.info(f"PyTorch推理时间: {inference_time:.4f}秒")
    return output_tensor, inference_time

def process_with_onnx(input_tensor, onnx_model, args, logger):
    """使用ONNX模型处理图像张量"""
    # 将PyTorch张量转换为NumPy数组
    if args.rgb_mode:
        # RGB模式需要分别处理各个通道
        channels = []
        start_time = time.time()
        for c in range(3):
            channel_input = input_tensor[:, c:c+1, :, :].numpy()
            ort_inputs = {onnx_model['input_name']: channel_input}
            channel_output = onnx_model['session'].run([onnx_model['output_name']], ort_inputs)[0]
            channels.append(channel_output)
        
        # 合并通道
        output_array = np.concatenate(channels, axis=1)
        inference_time = time.time() - start_time
    else:
        # 单通道处理
        input_array = input_tensor.numpy()
        ort_inputs = {onnx_model['input_name']: input_array}
        
        # ONNX推理
        start_time = time.time()
        output_array = onnx_model['session'].run([onnx_model['output_name']], ort_inputs)[0]
        inference_time = time.time() - start_time
    
    logger.info(f"ONNX推理时间: {inference_time:.4f}秒")
    
    # 将NumPy数组转换回PyTorch张量
    output_tensor = torch.from_numpy(output_array)
    return output_tensor, inference_time

def postprocess_tensor(output_tensor, args, logger):
    """将输出张量转换为PIL图像"""
    # 检查输出形状
    logger.info(f"输出张量形状: {output_tensor.shape}")
    
    # 后处理
    if args.ultrasound_mode or (not args.rgb_mode and output_tensor.shape[1] == 1):
        # 超声模式输出或灰度输出：单通道灰度图像
        output_img = output_tensor.squeeze().cpu().numpy()
        output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_img, mode='L')
    else:
        # RGB模式输出：三通道彩色图像
        output_img = output_tensor.squeeze().cpu()
        output_img = TF.to_pil_image(output_img)
    
    # 输出图像尺寸
    sr_width, sr_height = output_img.size
    logger.info(f"超分辨率图像尺寸: {sr_width}x{sr_height}")
    
    return output_img

def calculate_metrics(input_tensor, output_tensor, reference_tensor, logger):
    """计算PSNR和SSIM指标"""
    # 确保设备一致
    device = output_tensor.device
    reference_tensor = reference_tensor.to(device)
    
    # 确保尺寸匹配
    if reference_tensor.shape != output_tensor.shape:
        logger.warning(f"参考图像形状 {reference_tensor.shape} 与输出形状 {output_tensor.shape} 不匹配，将调整参考图像尺寸")
        reference_tensor = torch.nn.functional.interpolate(
            reference_tensor, 
            size=(output_tensor.shape[2], output_tensor.shape[3]),
            mode='bicubic',
            align_corners=False
        )
    
    # 计算度量指标
    with torch.no_grad():
        psnr_value = calculate_psnr(output_tensor, reference_tensor).item()
        ssim_value = calculate_ssim(output_tensor, reference_tensor).item()
    
    logger.info(f"PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
    
    # 计算提升（与输入LR相比）
    if input_tensor.shape[2:] != output_tensor.shape[2:]:
        # 将输入图像上采样到与输出相同的尺寸，用于比较
        upscaled_input = torch.nn.functional.interpolate(
            input_tensor, 
            size=(output_tensor.shape[2], output_tensor.shape[3]),
            mode='bicubic',
            align_corners=False
        )
        
        input_psnr = calculate_psnr(upscaled_input, reference_tensor).item()
        input_ssim = calculate_ssim(upscaled_input, reference_tensor).item()
        
        logger.info(f"提升 - PSNR: +{psnr_value - input_psnr:.2f} dB, SSIM: +{ssim_value - input_ssim:.4f}")
    
    return psnr_value, ssim_value

def process_image(image_path, model, device, args, logger, reference_path=None):
    """处理单个图像"""
    logger.info(f"处理图像: {image_path}")
    
    try:
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # 预处理图像
        input_tensor = preprocess_image(img, args, logger)
        
        # 根据模型类型选择处理方法
        if isinstance(model, dict) and 'session' in model:  # ONNX模型
            output_tensor, inference_time = process_with_onnx(input_tensor, model, args, logger)
        else:  # PyTorch模型
            output_tensor, inference_time = process_with_pytorch(input_tensor, model, device, args, logger)
        
        # 后处理为PIL图像
        output_img = postprocess_tensor(output_tensor, args, logger)
        
        # 如果提供了参考图像，计算质量指标
        metrics = None
        if reference_path and os.path.exists(reference_path):
            try:
                logger.info(f"加载参考图像: {reference_path}")
                ref_img = Image.open(reference_path).convert('RGB')
                
                # 根据模式处理参考图像
                if args.ultrasound_mode:
                    ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0)
                    ref_ycbcr = rgb_to_ycbcr(ref_tensor)
                    ref_tensor = ref_ycbcr[:, 0:1, :, :]
                elif args.rgb_mode:
                    ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0)
                else:
                    ref_tensor = transforms.Grayscale(num_output_channels=1)(ref_img)
                    ref_tensor = transforms.ToTensor()(ref_tensor).unsqueeze(0)
                
                metrics = calculate_metrics(input_tensor, output_tensor, ref_tensor, logger)
            except Exception as e:
                logger.error(f"计算质量指标时出错: {e}")
        
        return output_img, inference_time, metrics
    
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 0, None

def process_video(video_path, model, device, args, logger, reference_video=None):
    """处理视频文件"""
    logger.info(f"处理视频: {video_path}")
    
    try:
        # 打开输入视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 设置输出视频FPS
        output_fps = args.video_fps if args.video_fps > 0 else original_fps
        
        # 创建输出视频文件名
        if not args.no_save:
            # 创建输出目录
            os.makedirs(args.output, exist_ok=True)
            
            # 分析输入文件名
            video_name = os.path.basename(video_path)
            name, ext = os.path.splitext(video_name)
            
            # 构建输出文件路径
            output_path = os.path.join(args.output, f"{name}_x{args.scale}{ext}")
            
            # 创建VideoWriter对象
            sr_width = width * args.scale
            sr_height = height * args.scale
            
            # 确定输出视频编解码器
            fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (sr_width, sr_height))
            
            logger.info(f"输出视频: {output_path}")
            logger.info(f"输出视频尺寸: {sr_width}x{sr_height}, FPS: {output_fps}")
        
        # 打开参考视频（如果提供）
        ref_cap = None
        if reference_video and os.path.exists(reference_video):
            ref_cap = cv2.VideoCapture(reference_video)
            if not ref_cap.isOpened():
                logger.warning(f"无法打开参考视频文件: {reference_video}")
                ref_cap = None
            else:
                logger.info(f"加载参考视频: {reference_video}")
        
        # 进度信息
        logger.info(f"开始处理视频, 总帧数: {frame_count}")
        
        # 处理统计
        frame_idx = 0
        processed_frames = 0
        total_inference_time = 0
        total_psnr = 0
        total_ssim = 0
        
        # 创建进度条
        progress_bar = tqdm(total=frame_count, desc="处理视频帧", unit="帧")
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 是否跳过这一帧
            frame_idx += 1
            if args.skip_frames > 0 and frame_idx % (args.skip_frames + 1) != 1:
                progress_bar.update(1)
                continue
            
            # 读取参考帧（如果有）
            ref_frame = None
            if ref_cap is not None:
                ref_ret, ref_frame = ref_cap.read()
                if not ref_ret:
                    logger.warning("参考视频帧不足，停止质量评估")
                    ref_cap.release()
                    ref_cap = None
            
            # 将OpenCV的BGR图像转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # 预处理图像
            input_tensor = preprocess_image(img, args, logger)
            
            # 根据模型类型选择处理方法
            if isinstance(model, dict) and 'session' in model:  # ONNX模型
                output_tensor, inference_time = process_with_onnx(input_tensor, model, args, logger)
            else:  # PyTorch模型
                output_tensor, inference_time = process_with_pytorch(input_tensor, model, device, args, logger)
            
            # 累加推理时间
            total_inference_time += inference_time
            
            # 后处理为PIL图像
            output_img = postprocess_tensor(output_tensor, args, logger)
            
            # 如果有参考帧，计算质量指标
            if ref_frame is not None:
                ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
                ref_img = Image.fromarray(ref_rgb)
                
                # 根据模式处理参考图像
                if args.ultrasound_mode:
                    ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0)
                    ref_ycbcr = rgb_to_ycbcr(ref_tensor)
                    ref_tensor = ref_ycbcr[:, 0:1, :, :]
                elif args.rgb_mode:
                    ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0)
                else:
                    ref_tensor = transforms.Grayscale(num_output_channels=1)(ref_img)
                    ref_tensor = transforms.ToTensor()(ref_tensor).unsqueeze(0)
                
                try:
                    psnr, ssim = calculate_metrics(input_tensor, output_tensor, ref_tensor, logger)
                    total_psnr += psnr
                    total_ssim += ssim
                except Exception as e:
                    logger.error(f"计算质量指标时出错: {e}")
            
            # 显示结果
            if args.show:
                cv2_output = np.array(output_img)
                # 转换回BGR颜色空间（如果是RGB）
                if args.rgb_mode or (not args.ultrasound_mode and output_tensor.shape[1] == 3):
                    cv2_output = cv2.cvtColor(cv2_output, cv2.COLOR_RGB2BGR)
                # 显示图像
                cv2.imshow("Super Resolution", cv2_output)
                key = cv2.waitKey(1)
                if key == 27:  # ESC键
                    logger.info("用户中断处理")
                    break
            
            # 保存结果
            if not args.no_save:
                # 转换为OpenCV格式
                cv2_output = np.array(output_img)
                # 转换回BGR颜色空间（如果是RGB）
                if args.rgb_mode or (not args.ultrasound_mode and output_tensor.shape[1] == 3):
                    cv2_output = cv2.cvtColor(cv2_output, cv2.COLOR_RGB2BGR)
                # 写入视频
                out.write(cv2_output)
            
            processed_frames += 1
            progress_bar.update(1)
        
        # 关闭资源
        progress_bar.close()
        cap.release()
        if ref_cap is not None:
            ref_cap.release()
        if not args.no_save:
            out.release()
        if args.show:
            cv2.destroyAllWindows()
        
        # 计算和显示最终结果
        if processed_frames > 0:
            avg_inference_time = total_inference_time / processed_frames
            logger.info(f"视频处理完成，共处理 {processed_frames} 帧")
            logger.info(f"平均每帧推理时间: {avg_inference_time:.4f}秒")
            logger.info(f"平均FPS: {1.0/avg_inference_time:.2f}")
            
            if total_psnr > 0 and total_ssim > 0:
                avg_psnr = total_psnr / processed_frames
                avg_ssim = total_ssim / processed_frames
                logger.info(f"平均PSNR: {avg_psnr:.2f}dB, 平均SSIM: {avg_ssim:.4f}")
        
    except Exception as e:
        logger.error(f"处理视频失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def batch_process_images(input_dir, model, device, args, logger, reference_dir=None):
    """批量处理目录中的所有图像"""
    # 查找所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    
    if not image_files:
        logger.error(f"在 {input_dir} 中没有找到图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 创建输出目录
    if not args.no_save:
        os.makedirs(args.output, exist_ok=True)
    
    # 处理统计
    total_time = 0
    processed_count = 0
    total_psnr = 0
    total_ssim = 0
    
    # 处理每个图像
    for img_file in image_files:
        # 查找对应的参考图像（如果有）
        ref_image = None
        if reference_dir:
            ref_path = Path(reference_dir) / img_file.name
            if ref_path.exists():
                ref_image = str(ref_path)
        
        # 处理图像
        output_img, inference_time, metrics = process_image(
            str(img_file), model, device, args, logger, ref_image
        )
        
        if output_img:
            # 保存结果
            if not args.no_save:
                output_file = os.path.join(args.output, f"{img_file.stem}_x{args.scale}{img_file.suffix}")
                output_img.save(output_file)
                logger.info(f"结果已保存到: {output_file}")
            
            # 显示结果
            if args.show:
                output_img.show()
            
            # 累加统计
            total_time += inference_time
            processed_count += 1
            
            # 累加质量指标
            if metrics:
                psnr, ssim = metrics
                total_psnr += psnr
                total_ssim += ssim
    
    # 计算平均值
    if processed_count > 0:
        avg_time = total_time / processed_count
        logger.info(f"批处理完成，处理了 {processed_count} 张图像")
        logger.info(f"平均每张图像推理时间: {avg_time:.4f}秒")
        
        if total_psnr > 0 and total_ssim > 0:
            avg_psnr = total_psnr / processed_count
            avg_ssim = total_ssim / processed_count
            logger.info(f"平均PSNR: {avg_psnr:.2f}dB, 平均SSIM: {avg_ssim:.4f}")

# 添加简易GUI类
def create_gui():
    """创建并启动简易的图形界面"""
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    from threading import Thread
    
    root = tk.Tk()
    root.title("EchoSRNet图像超分辨率")
    root.geometry("700x500")
    root.resizable(True, True)
    
    # 存储模型列表
    pretrained_models = list_pretrained_models()
    
    # 存储用户选择
    input_path_var = tk.StringVar()
    output_path_var = tk.StringVar()
    model_var = tk.StringVar()
    scale_var = tk.IntVar(value=2)
    ultrasound_var = tk.BooleanVar(value=False)
    rgb_var = tk.BooleanVar(value=False)
    cpu_var = tk.BooleanVar(value=False)
    show_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value="就绪")
    
    # 创建日志区域
    log_text = tk.Text(root, height=10, width=80)
    log_text.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
    
    # 添加滚动条
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=log_text.yview)
    scrollbar.grid(row=7, column=3, sticky="ns")
    log_text.config(yscrollcommand=scrollbar.set)
    
    # 创建日志处理器
    class TextHandler(logging.Handler):
        def __init__(self, text):
            logging.Handler.__init__(self)
            self.text = text
            
        def emit(self, record):
            msg = self.format(record)
            def append():
                self.text.configure(state='normal')
                self.text.insert(tk.END, msg + '\n')
                self.text.configure(state='disabled')
                self.text.yview(tk.END)
            self.text.after(0, append)
    
    # 设置日志
    logger = logging.getLogger('infer_gui')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = TextHandler(log_text)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 浏览输入文件
    def browse_input():
        filetypes = [("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        filepath = filedialog.askopenfilename(title="选择输入图像", filetypes=filetypes)
        if filepath:
            input_path_var.set(filepath)
            # 自动设置输出路径
            if not output_path_var.get():
                input_file = Path(filepath)
                output_file = str(input_file.parent / f"{input_file.stem}_SR{input_file.suffix}")
                output_path_var.set(output_file)
    
    # 浏览输出文件
    def browse_output():
        filetypes = [("图像文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        filepath = filedialog.asksaveasfilename(title="保存输出图像", filetypes=filetypes, defaultextension=".png")
        if filepath:
            output_path_var.set(filepath)
    
    # 当选择预训练模型时
    def on_model_select(event):
        selection = model_listbox.curselection()
        if selection:
            model_name = model_listbox.get(selection[0])
            model_var.set(model_name)
            
            # 自动设置超声模式
            if 'us' in model_name.lower():
                ultrasound_var.set(True)
            
            # 自动设置缩放因子
            if 'scale4' in model_name.lower():
                scale_var.set(4)
            else:
                scale_var.set(2)
    
    # 处理图像
    def process_image_thread():
        try:
            # 检查输入
            if not input_path_var.get():
                messagebox.showerror("错误", "请选择输入图像")
                return
                
            if not output_path_var.get():
                messagebox.showerror("错误", "请选择输出路径")
                return
                
            if not model_var.get():
                messagebox.showerror("错误", "请选择预训练模型")
                return
            
            # 禁用处理按钮
            process_button.config(state=tk.DISABLED)
            status_var.set("处理中...")
            
            # 创建参数对象
            class Args:
                pass
                
            args = Args()
            args.model = os.path.join(current_dir, 'trained', model_var.get())
            args.input = input_path_var.get()
            args.output = output_path_var.get()
            args.scale = scale_var.get()
            args.ultrasound_mode = ultrasound_var.get()
            args.rgb_mode = rgb_var.get()
            args.cpu = cpu_var.get()
            args.show = show_var.get()
            args.no_save = False
            args.verbose = True
            args.m = 5
            args.int_features = 16
            args.feature_size = 256
            args.onnx_model = None
            args.use_speckle_filter = None
            args.use_edge_enhancer = None
            args.use_signal_enhancer = None
            args.reference = None
            
            # 设置设备
            if args.cpu:
                device = torch.device('cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"使用设备: {device}")
            
            # 加载模型
            model = load_pytorch_model(args, device, logger)
            
            # 处理图像
            input_path = Path(args.input)
            output_img, inference_time, metrics = process_image(
                str(input_path), model, device, args, logger, None
            )
            
            if output_img:
                # 保存结果
                # 检查输出是否是完整的文件路径（包含扩展名），而不是目录路径
                if os.path.splitext(args.output)[1]:  # 有扩展名，视为文件
                    output_file = args.output
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                else:  # 没有扩展名，视为目录
                    # 确保输出目录存在
                    os.makedirs(args.output, exist_ok=True)
                    # 从输入文件名构建输出文件名
                    output_file = os.path.join(args.output, f"{input_path.stem}_x{args.scale}{input_path.suffix}")
                
                output_img.save(output_file)
                logger.info(f"结果已保存到: {output_file}")
                
                # 显示结果
                if args.show:
                    output_img.show()
                
                status_var.set(f"处理完成 (推理时间: {inference_time:.2f}秒)")
            
        except Exception as e:
            logger.error(f"处理出错: {str(e)}")
            status_var.set(f"处理出错")
            messagebox.showerror("错误", f"处理图像时出错: {str(e)}")
        finally:
            # 重新启用处理按钮
            process_button.config(state=tk.NORMAL)
    
    def process_image():
        thread = Thread(target=process_image_thread)
        thread.daemon = True
        thread.start()
    
    # 创建UI元素
    # 输入图像
    ttk.Label(root, text="输入图像:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(root, textvariable=input_path_var, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(root, text="浏览...", command=browse_input).grid(row=0, column=2, padx=5, pady=5)
    
    # 输出路径
    ttk.Label(root, text="输出路径:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Entry(root, textvariable=output_path_var, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(root, text="浏览...", command=browse_output).grid(row=1, column=2, padx=5, pady=5)
    
    # 预训练模型列表
    ttk.Label(root, text="预训练模型:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    model_frame = ttk.Frame(root)
    model_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
    
    model_listbox = tk.Listbox(model_frame, height=5, width=50)
    model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    model_scrollbar = ttk.Scrollbar(model_frame, orient="vertical", command=model_listbox.yview)
    model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    model_listbox.config(yscrollcommand=model_scrollbar.set)
    
    # 填充模型列表
    for model in pretrained_models:
        model_listbox.insert(tk.END, model)
    
    model_listbox.bind('<<ListboxSelect>>', on_model_select)
    
    # 缩放因子
    ttk.Label(root, text="缩放因子:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
    scale_frame = ttk.Frame(root)
    scale_frame.grid(row=3, column=1, padx=5, pady=5, sticky="w")
    ttk.Radiobutton(scale_frame, text="2x", variable=scale_var, value=2).pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(scale_frame, text="4x", variable=scale_var, value=4).pack(side=tk.LEFT, padx=10)
    
    # 选项
    options_frame = ttk.LabelFrame(root, text="选项")
    options_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
    
    ttk.Checkbutton(options_frame, text="超声模式", variable=ultrasound_var).grid(row=0, column=0, padx=10, pady=5, sticky="w")
    ttk.Checkbutton(options_frame, text="RGB模式", variable=rgb_var).grid(row=0, column=1, padx=10, pady=5, sticky="w")
    ttk.Checkbutton(options_frame, text="CPU模式", variable=cpu_var).grid(row=0, column=2, padx=10, pady=5, sticky="w")
    ttk.Checkbutton(options_frame, text="显示结果", variable=show_var).grid(row=0, column=3, padx=10, pady=5, sticky="w")
    
    # 状态栏
    ttk.Label(root, textvariable=status_var).grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="w")
    
    # 处理按钮
    process_button = ttk.Button(root, text="开始处理", command=process_image)
    process_button.grid(row=6, column=0, columnspan=3, padx=5, pady=10)
    
    # 设置网格权重
    root.columnconfigure(1, weight=1)
    root.rowconfigure(7, weight=1)
    
    # 启动GUI
    root.mainloop()

# 修改main函数，支持GUI模式
def main():
    # 解析参数
    args = parse_args()
    
    # 检查是否要启动GUI
    if args.gui:
        try:
            # 尝试导入gui模块并启动GUI界面
            import gui
            gui.main()
            return
        except ImportError:
            print("无法导入GUI模块，请确保gui.py在同一目录下")
            print("继续使用命令行模式...")
    
    # 如果只是列出预训练模型
    if args.list_pretrained:
        print("可用的预训练模型:")
        for model in list_pretrained_models():
            print(f"  - {model}")
        return
    
    # 命令行模式的处理逻辑
    # 设置日志
    logger = setup_logger(args.verbose)
    
    # 处理预训练模型选项
    if args.use_pretrained:
        pretrained_models = list_pretrained_models()
        if not pretrained_models:
            logger.error("找不到预训练模型，请确保trained文件夹存在并包含.pth文件")
            return
        
        if args.use_pretrained not in pretrained_models:
            logger.error(f"指定的预训练模型 '{args.use_pretrained}' 不存在")
            logger.info("可用的预训练模型:")
            for model in pretrained_models:
                logger.info(f"  - {model}")
            return
        
        # 设置模型路径
        args.model = os.path.join(current_dir, 'trained', args.use_pretrained)
        logger.info(f"使用预训练模型: {args.use_pretrained}")
        
        # 根据模型名称设置相关参数
        if 'us' in args.use_pretrained.lower() and not args.ultrasound_mode:
            logger.info("检测到超声模型，自动启用超声模式")
            args.ultrasound_mode = True
            
        # 根据模型名称设置EchoSPANNet特有参数
        if 'echospannet' in args.use_pretrained.lower():
            logger.info("检测到EchoSPANNet模型")
            if not hasattr(args, 'span_feature_channels') or args.span_feature_channels == 48:
                args.span_feature_channels = 48
                logger.info(f"设置EchoSPANNet特征通道数: {args.span_feature_channels}")
            
            if not hasattr(args, 'span_num_blocks') or args.span_num_blocks == 6:
                args.span_num_blocks = 6
                logger.info(f"设置EchoSPANNet块数量: {args.span_num_blocks}")
            
            # 根据文件名判断是否启用傅里叶处理
            args.use_fourier = True
            logger.info("启用傅里叶变换处理")
    
    # 确保至少指定了一种模型
    if not args.model and not args.onnx_model and not args.use_pretrained:
        logger.error("必须指定模型路径，使用--model、--onnx_model或--use_pretrained")
        return
    
    # 设置设备
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    if args.model:
        model = load_pytorch_model(args, device, logger)
        
        # 测试推理速度
        if args.test_speed:
            test_inference_speed(model, args, device, logger)
    elif args.onnx_model:
        model = load_onnx_model(args, logger)
    
    # 获取输入路径
    input_path = Path(args.input)
    
    # 确定引用路径
    reference_path = args.reference if args.reference else None
    
    # 基于输入类型选择处理方法
    if args.video or (input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']):
        # 处理视频
        process_video(str(input_path), model, device, args, logger, reference_path)
    elif input_path.is_file():
        # 处理单个图像
        output_img, inference_time, metrics = process_image(
            str(input_path), model, device, args, logger, reference_path
        )
        
        if output_img:
            # 保存结果
            if not args.no_save:
                # 检查输出是否是完整的文件路径（包含扩展名），而不是目录路径
                if os.path.splitext(args.output)[1]:  # 有扩展名，视为文件
                    output_file = args.output
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                else:  # 没有扩展名，视为目录
                    # 确保输出目录存在
                    os.makedirs(args.output, exist_ok=True)
                    # 从输入文件名构建输出文件名
                    output_file = os.path.join(args.output, f"{input_path.stem}_x{args.scale}{input_path.suffix}")
                
                output_img.save(output_file)
                logger.info(f"结果已保存到: {output_file}")
            
            # 显示结果
            if args.show:
                output_img.show()
    elif input_path.is_dir():
        # 处理目录中的所有图像
        batch_process_images(str(input_path), model, device, args, logger, reference_path)
    else:
        logger.error(f"输入路径不存在: {args.input}")

# 添加测试推理速度的函数
def test_inference_speed(model, args, device, logger):
    """测试模型的推理速度
    
    Args:
        model: 模型
        args: 参数设置
        device: 设备
        logger: 日志记录器
    """
    logger.info(f"开始推理速度测试，迭代次数: {args.test_iterations}")
    
    # 确保模型在正确设备并处于评估模式
    model = model.to(device)
    model.eval()
    
    # 获取输入维度
    input_shape = (1, 1, 64, 64)  # 批大小, 通道数, 高度, 宽度
    if args.rgb_mode:
        input_shape = (1, 3, 64, 64)
    
    logger.info(f"使用输入形状: {input_shape}")
    
    # 创建随机输入张量
    input_tensor = torch.rand(input_shape, device=device)
    
    # 预热
    logger.info("预热 GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # 测试模型速度
    logger.info("测试模型推理速度...")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(args.test_iterations):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # 计算每次推理平均时间(毫秒)和FPS
    avg_time_ms = total_time * 1000 / args.test_iterations
    fps = args.test_iterations / total_time
    
    # 输出结果
    logger.info(f"推理速度测试结果:")
    logger.info(f"总测试时间: {total_time:.2f} 秒")
    logger.info(f"平均推理时间: {avg_time_ms:.2f} ms/image")
    logger.info(f"推理FPS: {fps:.2f}")
    
    return avg_time_ms, fps

if __name__ == '__main__':
    main() 