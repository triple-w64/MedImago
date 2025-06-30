import os
import time
import cv2
import numpy as np
import math
import sys
import torch
import torch.nn as nn
from PIL import Image
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QFileDialog, QFrame, QSplitter, 
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QProgressBar,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QScrollArea,
    QGraphicsLineItem, QGraphicsTextItem, QApplication, QDialog,
    QSlider, QGridLayout, QDialogButtonBox, QRubberBand, QDockWidget,
    QMainWindow
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QWheelEvent, QCursor, QPainter, QBrush, QColor, QPen, QKeySequence
from PyQt5.QtWidgets import QShortcut

# 添加Model目录到路径
model_dir = os.path.join(os.path.dirname(__file__), 'Model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# 导入网络结构
try:
    from echospan_net import EchoSPANNet
    from FPRAEcho import IPGGNNEchoSPANNet
except ImportError as e:
    print(f"Warning: Could not import network modules: {e}")
    EchoSPANNet = None
    IPGGNNEchoSPANNet = None

# 添加窗位窗宽调整对话框
class WindowLevelDialog(QDialog):
    def __init__(self, parent=None, window_level=128, window_width=255):
        super().__init__(parent)
        self.setWindowTitle("窗位窗宽调整")
        self.setMinimumWidth(400)
        
        # 保存父对象和原始图像
        self.parent = parent
        self.original_image = parent.image.copy() if parent.image is not None else None
        
        layout = QGridLayout(self)
        
        # 窗位滑块
        layout.addWidget(QLabel("窗位:"), 0, 0)
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setMinimum(0)
        self.level_slider.setMaximum(255)
        self.level_slider.setValue(window_level)
        layout.addWidget(self.level_slider, 0, 1)
        self.level_value = QLabel(str(window_level))
        layout.addWidget(self.level_value, 0, 2)
        
        # 窗宽滑块
        layout.addWidget(QLabel("窗宽:"), 1, 0)
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(255)
        self.width_slider.setValue(window_width)
        layout.addWidget(self.width_slider, 1, 1)
        self.width_value = QLabel(str(window_width))
        layout.addWidget(self.width_value, 1, 2)
        
        # 添加按钮
        self.button_box = QDialogButtonBox()
        self.apply_button = self.button_box.addButton("应用", QDialogButtonBox.ApplyRole)
        self.button_box.addButton(QDialogButtonBox.Ok)
        self.button_box.addButton(QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box, 2, 0, 1, 3)
        
        # 连接信号
        self.level_slider.valueChanged.connect(self.update_level_value)
        self.level_slider.valueChanged.connect(self.preview_changes)
        self.width_slider.valueChanged.connect(self.update_width_value)
        self.width_slider.valueChanged.connect(self.preview_changes)
        self.apply_button.clicked.connect(self.apply_changes)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        # 初始预览
        self.preview_changes()
        
    def update_level_value(self, value):
        self.level_value.setText(str(value))
        
    def update_width_value(self, value):
        self.width_value.setText(str(value))
        
    def get_values(self):
        return self.level_slider.value(), self.width_slider.value()
    
    def preview_changes(self):
        """实时预览窗位窗宽调整"""
        if self.original_image is None:
            return
            
        try:
            level, width = self.get_values()
            
            # 计算上下阈值
            min_val = max(0, level - width // 2)
            max_val = min(255, level + width // 2)
            
            # 复制原图以保留原始数据
            adjusted_img = self.original_image.copy()
            
            # 如果是彩色图像，分别处理每个通道
            if len(adjusted_img.shape) == 3 and adjusted_img.shape[2] >= 3:
                for i in range(3):  # 处理BGR三个通道
                    # 限制在阈值范围内
                    adjusted_img[:, :, i] = np.clip(adjusted_img[:, :, i], min_val, max_val)
                    # 重新映射到0-255
                    if max_val > min_val:  # 避免除零错误
                        adjusted_img[:, :, i] = ((adjusted_img[:, :, i] - min_val) / (max_val - min_val)) * 255
            else:
                # 灰度图像处理
                adjusted_img = np.clip(adjusted_img, min_val, max_val)
                if max_val > min_val:  # 避免除零错误
                    adjusted_img = ((adjusted_img - min_val) / (max_val - min_val)) * 255
            
            # 确保结果是uint8类型
            adjusted_img = adjusted_img.astype(np.uint8)
            
            # 显示结果
            self.parent.display_image(self.parent.original_image_label, adjusted_img)
            
        except Exception as e:
            print(f"预览窗位窗宽调整失败: {str(e)}")
    
    def apply_changes(self):
        """应用当前设置但不关闭对话框"""
        if self.original_image is None:
            return
            
        try:
            level, width = self.get_values()
            
            # 计算上下阈值
            min_val = max(0, level - width // 2)
            max_val = min(255, level + width // 2)
            
            # 复制原图以保留原始数据
            adjusted_img = self.original_image.copy()
            
            # 如果是彩色图像，分别处理每个通道
            if len(adjusted_img.shape) == 3 and adjusted_img.shape[2] >= 3:
                for i in range(3):  # 处理BGR三个通道
                    # 限制在阈值范围内
                    adjusted_img[:, :, i] = np.clip(adjusted_img[:, :, i], min_val, max_val)
                    # 重新映射到0-255
                    if max_val > min_val:  # 避免除零错误
                        adjusted_img[:, :, i] = ((adjusted_img[:, :, i] - min_val) / (max_val - min_val)) * 255
            else:
                # 灰度图像处理
                adjusted_img = np.clip(adjusted_img, min_val, max_val)
                if max_val > min_val:  # 避免除零错误
                    adjusted_img = ((adjusted_img - min_val) / (max_val - min_val)) * 255
            
            # 确保结果是uint8类型
            adjusted_img = adjusted_img.astype(np.uint8)
            
            # 显示结果并更新父组件的图像
            self.parent.display_image(self.parent.original_image_label, adjusted_img)
            self.parent.image = adjusted_img.copy()
            self.original_image = adjusted_img.copy()  # 更新对话框中的原始图像
            
            self.parent.status_bar.showMessage(f"窗位窗宽调整已应用: 窗位={level}, 窗宽={width}")
        except Exception as e:
            self.parent.status_bar.showMessage(f"窗位窗宽调整失败: {str(e)}")
    
    def accept(self):
        """确定按钮处理"""
        # 应用当前设置
        level, width = self.get_values()
        
        # 确保父组件的图像已更新为最终结果
        # (已在preview_changes中处理)
        
        self.parent.status_bar.showMessage(f"窗位窗宽调整已完成: 窗位={level}, 窗宽={width}")
        super().accept()

# 添加亮度对比度调整对话框
class BrightnessContrastDialog(QDialog):
    def __init__(self, parent=None, brightness=0, contrast=0):
        super().__init__(parent)
        self.setWindowTitle("亮度/对比度调整")
        self.setMinimumWidth(400)
        
        # 保存父对象和原始图像
        self.parent = parent
        self.original_image = parent.image.copy() if parent.image is not None else None
        
        layout = QGridLayout(self)
        
        # 亮度滑块
        layout.addWidget(QLabel("亮度:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(brightness)
        layout.addWidget(self.brightness_slider, 0, 1)
        self.brightness_value = QLabel(str(brightness))
        layout.addWidget(self.brightness_value, 0, 2)
        
        # 对比度滑块
        layout.addWidget(QLabel("对比度:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(contrast)
        layout.addWidget(self.contrast_slider, 1, 1)
        self.contrast_value = QLabel(str(contrast))
        layout.addWidget(self.contrast_value, 1, 2)
        
        # 添加按钮
        self.button_box = QDialogButtonBox()
        self.apply_button = self.button_box.addButton("应用", QDialogButtonBox.ApplyRole)
        self.button_box.addButton(QDialogButtonBox.Ok)
        self.button_box.addButton(QDialogButtonBox.Cancel)
        layout.addWidget(self.button_box, 2, 0, 1, 3)
        
        # 连接信号
        self.brightness_slider.valueChanged.connect(self.update_brightness_value)
        self.brightness_slider.valueChanged.connect(self.preview_changes)
        self.contrast_slider.valueChanged.connect(self.update_contrast_value)
        self.contrast_slider.valueChanged.connect(self.preview_changes)
        self.apply_button.clicked.connect(self.apply_changes)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        # 初始预览
        self.preview_changes()
        
    def update_brightness_value(self, value):
        self.brightness_value.setText(str(value))
        
    def update_contrast_value(self, value):
        self.contrast_value.setText(str(value))
        
    def get_values(self):
        return self.brightness_slider.value(), self.contrast_slider.value()
        
    def preview_changes(self):
        """实时预览亮度对比度调整"""
        if self.original_image is None:
            return
            
        try:
            brightness, contrast = self.get_values()
            
            # 复制原图以保留原始数据
            adjusted_img = self.original_image.copy()
            
            # 计算亮度和对比度参数
            alpha = (contrast + 100) / 100.0  # 对比度因子
            beta = brightness  # 亮度因子
            
            # 应用变换: g(x) = alpha * f(x) + beta
            adjusted_img = cv2.convertScaleAbs(adjusted_img, alpha=alpha, beta=beta)
            
            # 显示结果
            self.parent.display_image(self.parent.original_image_label, adjusted_img)
            
        except Exception as e:
            print(f"预览亮度对比度调整失败: {str(e)}")
    
    def apply_changes(self):
        """应用当前设置但不关闭对话框"""
        if self.original_image is None:
            return
            
        try:
            brightness, contrast = self.get_values()
            
            # 复制原图以保留原始数据
            adjusted_img = self.original_image.copy()
            
            # 计算亮度和对比度参数
            alpha = (contrast + 100) / 100.0  # 对比度因子
            beta = brightness  # 亮度因子
            
            # 应用变换: g(x) = alpha * f(x) + beta
            adjusted_img = cv2.convertScaleAbs(adjusted_img, alpha=alpha, beta=beta)
            
            # 显示结果并更新父组件的图像
            self.parent.display_image(self.parent.original_image_label, adjusted_img)
            self.parent.image = adjusted_img.copy()
            self.original_image = adjusted_img.copy()  # 更新对话框中的原始图像
            
            self.parent.status_bar.showMessage(f"亮度对比度调整已应用: 亮度={brightness}, 对比度={contrast}")
        except Exception as e:
            self.parent.status_bar.showMessage(f"亮度对比度调整失败: {str(e)}")
    
    def accept(self):
        """确定按钮处理"""
        # 应用当前设置
        brightness, contrast = self.get_values()
        
        # 确保父组件的图像已更新为最终结果
        # (已在preview_changes中处理)
        
        self.parent.status_bar.showMessage(f"亮度对比度调整已完成: 亮度={brightness}, 对比度={contrast}")
        super().accept()

# 超分辨率模块
class SuperResolutionTab(QWidget):
    def __init__(self, parent, status_bar):
        super().__init__(parent)
        self.main_window = parent
        self.status_bar = status_bar
        
        # 对话框实例变量
        self.window_level_dialog = None
        self.brightness_contrast_dialog = None
        
        # 添加距离测量相关变量
        self.measure_mode = False
        self.measure_start_point = None
        self.measure_end_point = None
        self.measure_line = None
        self.measure_text = None
        self.pixel_scale_mm = 0.2  # 默认像素比例尺，单位为mm/pixel
        self.measurement_lines = []  # 存储所有测量线
        self.measurement_texts = []  # 存储所有测量文本
        
        # 添加量角器相关变量
        self.angle_mode = False  # 量角器模式标志
        self.angle_lines = []  # 存储角度线
        self.angle_texts = []  # 存储角度文本
        self.angle_points = []  # 存储角度的顶点和两个端点
        self.angle_current_line = 0  # 当前正在绘制的角度线（0或1）
        
        # 添加指针和手型工具相关变量
        self.pointer_mode = True  # 默认为指针模式
        self.hand_mode = False  # 手型工具模式
        
        # 添加视频相关的初始化
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.image = None
        self.image_path = None
        self.sr_result = None
        
        # 添加模型相关的初始化
        self.sr_model = None
        self.model_type = None  # "opencv" 或 "pytorch"
        self.device = None
        self._current_model = None
        
        # 主布局改为水平布局
        self.main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        # 创建中右两个部分的容器
        self.center_container = QWidget()
        self.right_container = QWidget()
        
        # 设置布局
        self.center_layout = QVBoxLayout(self.center_container)
        self.right_layout = QVBoxLayout(self.right_container)
        
        # 创建组件
        self.create_video_player()
        self.create_sr_controls()
        
        # 将两个容器添加到 QSplitter
        splitter.addWidget(self.center_container)
        splitter.addWidget(self.right_container)
        
        # 设置拉伸因子(可自由调整)
        splitter.setStretchFactor(0, 4)  # 视频播放器
        splitter.setStretchFactor(1, 6)  # SR控制

        # 添加到主布局
        self.main_layout.addWidget(splitter)

        # 设置快捷键
        self.setup_shortcuts()
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self._cleanup_model()
        except:
            pass
    
    def set_file_tree(self, file_tree):
        """设置共享的文件树"""
        self.file_tree = file_tree

    def setup_shortcuts(self):
        """设置快捷键"""
        # Ctrl+Z 撤销上一步测量
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_measurement)
        
        # Ctrl+1 切换到指针工具
        self.pointer_shortcut = QShortcut(QKeySequence("Ctrl+1"), self)
        self.pointer_shortcut.activated.connect(lambda: self.switch_tool("pointer"))
        
        # Ctrl+2 切换到手型工具
        self.hand_shortcut = QShortcut(QKeySequence("Ctrl+2"), self)
        self.hand_shortcut.activated.connect(lambda: self.switch_tool("hand"))
        
        # Ctrl+3 切换到距离测量工具
        self.distance_shortcut = QShortcut(QKeySequence("Ctrl+3"), self)
        self.distance_shortcut.activated.connect(lambda: self.switch_tool("distance"))
        
        # Ctrl+4 切换到角度测量工具
        self.angle_shortcut = QShortcut(QKeySequence("Ctrl+4"), self)
        self.angle_shortcut.activated.connect(lambda: self.switch_tool("angle"))
        
        # Ctrl+W 窗位窗宽调整
        self.window_level_shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        self.window_level_shortcut.activated.connect(self.adjust_window_level)
        
        # Ctrl+B 亮度对比度调整
        self.brightness_contrast_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self.brightness_contrast_shortcut.activated.connect(self.adjust_brightness_contrast)
        
        # Ctrl+R ROI提取
        self.roi_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self.roi_shortcut.activated.connect(self.extract_roi)

    def switch_tool(self, tool_name):
        """切换工具"""
        # 先禁用所有工具
        self.pointer_mode = False
        self.hand_mode = False
        self.measure_mode = False
        self.angle_mode = False
        
        # 手动禁用测量工具
        if hasattr(self, 'enable_distance_measure'):
            self.enable_distance_measure(False)
        if hasattr(self, 'enable_angle_measure'):
            self.enable_angle_measure(False)
        
        # 根据选择激活相应工具
        if tool_name == "pointer":
            self.pointer_mode = True
            if hasattr(self.image_viewer, 'set_tool'):
                self.image_viewer.set_tool("pointer")
            self.status_bar.showMessage("已切换到指针工具")
        elif tool_name == "hand":
            self.hand_mode = True
            if hasattr(self.image_viewer, 'set_tool'):
                self.image_viewer.set_tool("hand")
            self.status_bar.showMessage("已切换到手型工具")
        elif tool_name == "distance":
            self.enable_distance_measure(True)
            # 状态栏消息已在enable_distance_measure中设置
        elif tool_name == "angle":
            self.enable_angle_measure(True)
            # 状态栏消息已在enable_angle_measure中设置
        
        # 更新工具栏按钮选中状态（如果存在）
        if hasattr(self.main_window, 'pointer_action'):
            self.main_window.pointer_action.setChecked(self.pointer_mode)
        if hasattr(self.main_window, 'hand_action'):
            self.main_window.hand_action.setChecked(self.hand_mode)
        if hasattr(self.main_window, 'distance_measure_action'):
            self.main_window.distance_measure_action.setChecked(self.measure_mode)
        if hasattr(self.main_window, 'angle_measure_action'):
            self.main_window.angle_measure_action.setChecked(self.angle_mode)

    def undo_last_measurement(self):
        """撤销上一步测量"""
        # 检查是否有测量可以撤销
        if self.measurement_lines and self.measurement_texts:
            # 移除最后一个测量线和文本
            last_line = self.measurement_lines.pop()
            last_text = self.measurement_texts.pop()
            
            scene = self.image_viewer.scene()
            if scene:
                if last_line and last_line.scene() == scene:
                    scene.removeItem(last_line)
                if last_text and last_text.scene() == scene:
                    scene.removeItem(last_text)
                
                # 强制刷新视图
                self.image_viewer.viewport().update()
                self.status_bar.showMessage("已撤销上一步测量")
        elif self.angle_lines and self.angle_texts:
            # 移除最后一组角度线和文本
            # 通常有两条线和一个文本
            if len(self.angle_lines) >= 2:
                last_lines = self.angle_lines[-2:]
                self.angle_lines = self.angle_lines[:-2]
                
                scene = self.image_viewer.scene()
                if scene:
                    for line in last_lines:
                        if line and line.scene() == scene:
                            scene.removeItem(line)
            
            if self.angle_texts:
                last_text = self.angle_texts.pop()
                scene = self.image_viewer.scene()
                if scene and last_text and last_text.scene() == scene:
                    scene.removeItem(last_text)
            
            # 强制刷新视图
            self.image_viewer.viewport().update()
            self.status_bar.showMessage("已撤销上一步角度测量")
        else:
            self.status_bar.showMessage("没有可撤销的测量")

    def create_file_manager(self):
        """创建文件管理器"""
        # 使用QTreeWidget
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("文件")
        self.file_tree.itemDoubleClicked.connect(self.open_selected_file)
        
        # 添加文件树展开事件处理
        self.file_tree.itemExpanded.connect(self.on_item_expanded)
        
        # 创建水平布局放置按钮
        button_layout = QHBoxLayout()
        
        load_folder_btn = QPushButton("加载文件夹")
        load_folder_btn.clicked.connect(self.load_folder)
        
        connect_stream_btn = QPushButton("连接流媒体") 
        connect_stream_btn.clicked.connect(self.connect_streaming)
        
        button_layout.addWidget(load_folder_btn)
        button_layout.addWidget(connect_stream_btn)
        
        self.left_layout.addWidget(self.file_tree)
        self.left_layout.addLayout(button_layout)

    def connect_streaming(self):
        """连接虚拟摄像头流"""
        try:
            # 释放之前的视频捕获（如果存在）
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
                self.timer.stop()

            # 连接2号虚拟摄像头
            self.video_capture = cv2.VideoCapture(2)
            if not self.video_capture.isOpened():
                self.status_bar.showMessage("无法连接到2号虚拟摄像头!")
                return

            # 获取第一帧并显示
            ret, frame = self.video_capture.read()
            if ret:
                # 显示图像尺寸
                height, width = frame.shape[:2]
                self.status_bar.showMessage(f"已连接到虚拟摄像头，图像尺寸: {width}x{height}")
                
                # 转换为RGB格式并显示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                
                # 启用视频控制按钮
                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.replay_button.setEnabled(True)
                self.capture_button.setEnabled(True)
                self.freeze_button.setEnabled(True)
                
                # 重置freeze状态
                self.is_frozen = False
                self.freeze_button.setText("Freeze")
                
                # 开始播放
                self.timer.start(30)  # 30ms per frame
                self.play_button.setEnabled(False)
            else:
                self.status_bar.showMessage("无法读取摄像头画面!")
                
        except Exception as e:
            self.status_bar.showMessage(f"连接摄像头时出错: {str(e)}")

    def create_video_player(self):
        """创建视频播放器"""
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.center_layout.addWidget(self.video_label)

        # 视频控制按钮
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放")
        self.replay_button = QPushButton("重新播放")
        self.stop_button = QPushButton("停止")
        self.capture_button = QPushButton("捕获帧")
        self.freeze_button = QPushButton("冻结")
        
        self.play_button.clicked.connect(self.play_video)
        self.replay_button.clicked.connect(self.replay_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.capture_button.clicked.connect(self.capture_frame)
        self.freeze_button.clicked.connect(self.toggle_freeze)
        
        self.video_progress = QProgressBar()
        
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.replay_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.freeze_button)
        
        self.center_layout.addWidget(self.video_progress)
        self.center_layout.addLayout(controls_layout)
        
        # 初始化freeze状态
        self.is_frozen = False

    def load_folder(self):
        """加载文件夹并创建树形结构"""
        try:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder_path:
                self.file_tree.clear()
                root = QTreeWidgetItem(self.file_tree, [os.path.basename(folder_path)])
                root.setData(0, Qt.UserRole, folder_path)
                
                # 添加目录内容
                self._add_directory_contents(root, folder_path)
                
                # 展开根节点
                root.setExpanded(True)
                self.status_bar.showMessage(f"Loaded folder: {folder_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading folder: {str(e)}")

    def _add_directory_contents(self, parent_item, parent_path):
        """优化目录内容加载，改用批量加载模式"""
        try:
            items = os.listdir(parent_path)
            # 先收集所有项目，然后批量添加
            dir_items = []
            file_items = []
            
            for item in items:
                if item.startswith('.'):  # 跳过隐藏文件
                    continue
                    
                full_path = os.path.join(parent_path, item)
                if os.path.isdir(full_path):
                    dir_item = QTreeWidgetItem(None, [item])
                    dir_item.setData(0, Qt.UserRole, full_path)
                    dir_items.append((dir_item, full_path))
                # 只处理视频和图像文件
                elif item.lower().endswith(('.png', '.jpg', '.bmp', '.mp4', '.avi', '.mkv')):
                    file_item = QTreeWidgetItem(None, [item])
                    file_item.setData(0, Qt.UserRole, full_path)
                    file_items.append(file_item)
            
            # 批量添加目录
            for dir_item, full_path in dir_items:
                parent_item.addChild(dir_item)
                # 添加展开监听器，确保可以打开下级文件夹
                dir_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            
            # 批量添加文件
            for file_item in file_items:
                parent_item.addChild(file_item)
                
        except Exception as e:
            self.status_bar.showMessage(f"Error loading directory contents: {str(e)}")

    def load_image(self, file_path=None):
        """加载图片文件"""
        try:
            if not file_path:
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Open Image",
                    "",
                    "Images (*.png *.jpg *.bmp)"
                )
            if file_path:
                # 使用IMREAD_UNCHANGED提高加载速度，保留原始图像格式
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    # 使用异步处理图像显示
                    self.status_bar.showMessage("Processing image...")
                    self.image = image
                    self.image_path = file_path
                    
                    # 转换为RGB格式并显示在video_label中
                    self._display_image_in_label(image, self.video_label)
                    
                    # 启用捕获按钮
                    self.capture_button.setEnabled(True)
                    
                    self.status_bar.showMessage("Image loaded successfully")
                else:
                    self.status_bar.showMessage("Failed to load image!")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}")

    def _display_image_in_label(self, image, label):
        """优化的图像显示逻辑，提高性能"""
        # 检查通道数
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # 灰度图像
            # 确保图像数据是连续的内存块
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
                
            h, w = image.shape if len(image.shape) == 2 else image.shape[:2]
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # 彩色图像
            if image.shape[2] == 4:  # RGBA图像
                # 转换为RGB格式
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:  # BGR图像
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 确保图像数据是连续的内存块    
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
                
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def display_image(self, label, image):
        """显示图像到标签或图像查看器"""
        if image is not None:
            if label == self.original_image_label:
                # 使用图像查看器显示图像
                h, w = image.shape[:2]
                if len(image.shape) == 2:  # 灰度图像
                    # 确保图像数据是连续的内存块
                    if not image.flags['C_CONTIGUOUS']:
                        image = np.ascontiguousarray(image)
                    bytes_per_line = w  # 一个像素一个字节
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                else:  # 彩色图像
                    if image.shape[2] == 4:  # RGBA图像
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    else:  # BGR图像
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 确保图像数据是连续的内存块
                    if not image.flags['C_CONTIGUOUS']:
                        image = np.ascontiguousarray(image)
                        
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                pixmap = QPixmap.fromImage(q_image)
                self.image_viewer.set_image(pixmap)
            else:
                # 普通标签显示图像
                self._display_image_in_label(image, label)

    def create_sr_controls(self):
        """创建超分辨率控制区域"""
        # 使用QSplitter使控件可以拉伸调整大小
        self.right_splitter = QSplitter(Qt.Vertical)
        
        # 图像显示组
        self.image_display_group = QGroupBox("图像显示")
        self.image_display_layout = QVBoxLayout(self.image_display_group)

        # 创建自定义的可缩放图像查看器
        self.image_viewer = ZoomableImageViewer()
        
        # 保留一个隐藏的标签用于兼容其他代码中对original_image_label的引用
        self.original_image_label = QLabel()
        self.original_image_label.setVisible(False)

        # 直接添加到布局，不使用dock widget
        self.image_display_layout.addWidget(self.image_viewer)
        self.image_display_layout.addWidget(self.original_image_label)
        
        # 控制组
        self.controls_group = QGroupBox("超分辨率控制")
        self.controls_layout = QVBoxLayout()

        # Scale Factor选择（水平布局）
        scale_layout = QHBoxLayout()
        self.scale_label = QLabel("放大倍数:")
        self.scale_spinner = QSpinBox()
        self.scale_spinner.setRange(2, 8)
        self.scale_spinner.setValue(2)
        scale_layout.addWidget(self.scale_label)
        scale_layout.addWidget(self.scale_spinner)
        scale_layout.addStretch()
        
        # Algorithm选择（水平布局）
        algo_layout = QHBoxLayout()
        self.algo_label = QLabel("SR算法:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Bilinear", "Bicubic", "EDSR", "ESPCN", "FSRCNN", "LAPSRN","IPGSPAN", "SPAN" ])
        algo_layout.addWidget(self.algo_label)
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addStretch()
        
        self.controls_layout.addLayout(scale_layout)
        self.controls_layout.addLayout(algo_layout)
        
        self.process_button = QPushButton("应用超分辨率")
        self.process_button.clicked.connect(self.apply_super_resolution)
        
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)

        self.controls_layout.addWidget(self.process_button)
        self.controls_layout.addWidget(self.save_button)

        self.controls_group.setLayout(self.controls_layout)
        
        # 添加到拆分器
        self.right_splitter.addWidget(self.image_display_group)
        self.right_splitter.addWidget(self.controls_group)
        
        # 设置初始大小比例
        self.right_splitter.setStretchFactor(0, 3)  # 图像显示区域
        self.right_splitter.setStretchFactor(1, 1)  # 控制区域
        
        # 添加到右布局
        self.right_layout.addWidget(self.right_splitter)

    def open_selected_file(self, item):
        """打开选中的文件"""
        file_path = item.data(0, Qt.UserRole)
        # 修改这里以支持 mkv
        if file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            self.load_video(file_path)
        else:
            self.load_image(file_path)

    def load_video(self, file_path):
        """加载视频文件"""
        if hasattr(self, 'video_capture') and self.video_capture is not None:
            self.video_capture.release()
            self.timer.stop()

        self.video_capture = cv2.VideoCapture(file_path)
        if self.video_capture.isOpened():
            # 清除之前的图像
            self.image = None
            self.sr_result = None
            self.save_button.setEnabled(False)
            
            # 更新进度条
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_progress.setMaximum(total_frames)
            self.video_progress.setValue(0)
            
            # 显示第一帧
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
            self.status_bar.showMessage(f"Video loaded: {file_path}")
            
            # 启用视频控制按钮
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.replay_button.setEnabled(True)
            self.capture_button.setEnabled(True)
        else:
            self.status_bar.showMessage("Failed to load video!")

    def play_video(self):
        """播放视频"""
        if self.video_capture is not None and self.video_capture.isOpened():
            self.timer.start(30)  # 30ms per frame
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.replay_button.setEnabled(True)

    def stop_video(self):
        """停止视频但保持当前帧"""
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        # 不再重置到第一帧,保持当前帧
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def update_frame(self):
        """优化的视频帧更新方法"""
        if self.video_capture is not None and not self.is_frozen:
            ret, frame = self.video_capture.read()
            if ret:
                # 显示图像尺寸
                height, width = frame.shape[:2]
                self.status_bar.showMessage(f"图像尺寸: {width}x{height}")
                
                # 直接调用优化的图像显示方法
                self._display_image_in_label(frame, self.video_label)

                current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                self.video_progress.setValue(current_frame)
            else:
                self.stop_video()

    def replay_video(self):
        """重新播放视频"""
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 将视频帧设置回起始位置
            self.play_video()

    def capture_frame(self):
        """捕获当前帧并进行处理"""
        if self.video_capture is not None:
            # 如果视频已冻结，使用上一帧，否则从视频中捕获新帧
            if self.is_frozen and hasattr(self, 'last_frame') and self.last_frame is not None:
                frame = self.last_frame.copy()
            else:
                ret, frame = self.video_capture.read()
                if not ret:
                    self.status_bar.showMessage("无法捕获视频帧!")
                    return
                self.last_frame = frame.copy()  # 保存当前帧用于冻结状态
            
            # 获取原始尺寸
            height, width = frame.shape[:2]
            
            # 裁切掉左侧160*480像素区域
            # 确保不会超出图像边界
            right_boundary = min(width, 160 + 480)
            if width > 160:
                cropped_frame = frame[:, 160:right_boundary].copy()
                crop_height, crop_width = cropped_frame.shape[:2]
                
                # 创建512*512的黑色画布
                canvas = np.zeros((512, 512, 3), dtype=np.uint8)
                
                # 计算居中放置的位置
                x_offset = (512 - crop_width) // 2
                y_offset = (512 - crop_height) // 2
                
                # 确保偏移量不为负
                x_offset = max(0, x_offset)
                y_offset = max(0, y_offset)
                
                # 计算实际可复制的区域大小
                copy_width = min(crop_width, 512 - x_offset)
                copy_height = min(crop_height, 512 - y_offset)
                
                # 将裁剪的图像复制到黑色画布的中心位置
                canvas[y_offset:y_offset+copy_height, x_offset:x_offset+copy_width] = \
                    cropped_frame[0:copy_height, 0:copy_width]
                
                # 保存处理后的图像
                self.image = canvas
                
                # 显示在original_image_label中
                self.display_image(self.original_image_label, self.image)
                self.status_bar.showMessage(f"已捕获当前帧并处理为512x512")
                
                # 启用超分辨率相关按钮
                self.process_button.setEnabled(True)
            else:
                self.status_bar.showMessage("图像太窄，无法按要求裁剪!")
        elif self.image is not None:
            # 如果没有视频但有图像(通过Open按钮加载的)，直接使用该图像
            # 将图像显示在original_image_label中
            self.display_image(self.original_image_label, self.image)
            self.status_bar.showMessage("图像已加载到处理区域")
            
            # 启用超分辨率相关按钮
            self.process_button.setEnabled(True)
        else:
            self.status_bar.showMessage("没有视频或图像可捕获!")

    def apply_super_resolution(self):
        if self.image is None:
            self.status_bar.showMessage("No image loaded!")
            return

        scale = self.scale_spinner.value()
        algorithm = self.algo_combo.currentText()
        
        # 获取原始图像尺寸
        orig_h, orig_w = self.image.shape[:2]
        
        start_time = time.time()

        if algorithm in ["EDSR", "ESPCN", "FSRCNN", "EchoSR", "LAPSRN", "SPAN", "IPGSPAN"]:
            model_loaded = self.load_dnn_sr_model(algorithm, scale)
            if not model_loaded:
                self.status_bar.showMessage(f"无法加载 {algorithm} 模型，请检查模型文件是否存在")
                return
                
            sr_image = self.run_dnn_super_resolution()
        elif algorithm in ["Bilinear", "Bicubic"]:
            sr_image = self.run_opencv_interpolation(algorithm, scale)
        else:
            sr_image = None

        end_time = time.time()
        if sr_image is not None:
            self.sr_result = sr_image  # 存储超分辨率结果
            # 显示重建结果
            self.display_image(self.original_image_label, sr_image)
            
            # 获取重建后的图像尺寸
            sr_h, sr_w = sr_image.shape[:2]
            
            elapsed_time = end_time - start_time
            # 同时显示原始尺寸和重建后的尺寸
            self.status_bar.showMessage(f"SR完成, 耗时: {elapsed_time:.2f}秒 - 原始尺寸: {orig_w}×{orig_h}, 重建尺寸: {sr_w}×{sr_h}")
            self.save_button.setEnabled(True)  # 启用保存按钮
        else:
            self.status_bar.showMessage("超分辨率重建失败!")
            self.save_button.setEnabled(False)

    def run_opencv_interpolation(self, method, scale):
        methods = {
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC
        }
        return cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=methods[method])

    def load_dnn_sr_model(self, method, scale):
        """优化模型加载过程"""
        # 传统OpenCV DNN模型
        opencv_models = {
            "EDSR": f"Model/EDSR_x{scale}.pb",
            "ESPCN": f"Model/ESPCN_x{scale}.pb",
            "FSRCNN": f"Model/FSRCNN_x{scale}.pb",
            "EchoSR": f"Model/EchoSR_x{scale}.pb",
            "LAPSRN": f"Model/LAPSRN_x{scale}.pb"
        }
        
        # PyTorch模型
        pytorch_models = {
            "SPAN": f"Model/echospannet_scale{scale}_best.pth",
            "IPGSPAN": f"Model/ipggnnechospannet_scale{scale}_best.pth"
        }
        
        # 检查模型是否已经加载过，避免重复加载相同的模型
        if hasattr(self, '_current_model') and self._current_model == (method, scale):
            return True
            
        # 清理之前的模型
        self._cleanup_model()
            
        # 记录当前加载的模型
        self._current_model = (method, scale)
        
        if method in opencv_models:
            return self._load_opencv_model(method, scale, opencv_models[method])
        elif method in pytorch_models:
            return self._load_pytorch_model(method, scale, pytorch_models[method])
        else:
            error_msg = f"不支持的模型类型: {method}"
            self.status_bar.showMessage(error_msg)
            return False
    
    def _load_opencv_model(self, method, scale, model_path):
        """加载OpenCV DNN模型"""
        # 显示模型路径，便于调试
        full_path = os.path.join(os.getcwd(), model_path)
        self.status_bar.showMessage(f"正在加载OpenCV模型: {full_path}")
        
        if not os.path.exists(full_path):
            error_msg = f"错误: 模型文件不存在: {full_path}"
            self.status_bar.showMessage(error_msg)
            print(error_msg)
            return False
            
        try:
            # 加载模型
            import cv2.dnn_superres
            self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr_model.readModel(full_path)
            self.sr_model.setModel(method.lower(), scale)
            self.model_type = "opencv"
            self.status_bar.showMessage(f"OpenCV模型加载成功: {method} x{scale}")
            return True
        except Exception as e:
            error_msg = f"模型加载错误: {str(e)}"
            self.status_bar.showMessage(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def _load_pytorch_model(self, method, scale, model_path):
        """加载PyTorch模型"""
        if method == "SPAN" and EchoSPANNet is None:
            error_msg = "EchoSPANNet网络模块未正确导入，无法加载SPAN模型"
            self.status_bar.showMessage(error_msg)
            return False
        elif method == "IPGSPAN" and IPGGNNEchoSPANNet is None:
            error_msg = "IPGGNNEchoSPANNet网络模块未正确导入，无法加载IPGSPAN模型"
            self.status_bar.showMessage(error_msg)
            return False
            
        # 显示模型路径，便于调试
        full_path = os.path.join(os.getcwd(), model_path)
        self.status_bar.showMessage(f"正在加载PyTorch模型: {full_path}")
        
        if not os.path.exists(full_path):
            error_msg = f"错误: 模型文件不存在: {full_path}"
            self.status_bar.showMessage(error_msg)
            print(error_msg)
            return False
            
        try:
            # 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 根据方法创建相应的网络
            if method == "SPAN":
                self.status_bar.showMessage(f"正在创建EchoSPANNet网络 (scale={scale})...")
                self.sr_model = EchoSPANNet(
                    scale=scale,
                    in_channels=1,
                    out_channels=1,
                    feature_channels=48,
                    num_blocks=6,
                    use_fourier=True,
                    img_range=1.0,
                    ultrasound_mode=True
                )
            elif method == "IPGSPAN":
                self.status_bar.showMessage(f"正在创建IPGGNNEchoSPANNet网络 (scale={scale})...")
                self.sr_model = IPGGNNEchoSPANNet(
                    scale=scale,
                    in_channels=1,
                    out_channels=1,
                    feature_channels=48,
                    num_blocks=6,
                    use_fourier=True,
                    img_range=1.0,
                    ultrasound_mode=True,
                    base_patch_size=64,
                    use_ipg=True,
                    max_degree=8
                )
            else:
                raise ValueError(f"不支持的PyTorch模型类型: {method}")
            
            # 加载预训练权重
            self.status_bar.showMessage(f"正在加载预训练权重...")
            checkpoint = torch.load(full_path, map_location=self.device)
            
            # 兼容不同的保存格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.status_bar.showMessage("检测到model_state_dict格式")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.status_bar.showMessage("检测到state_dict格式")
            else:
                state_dict = checkpoint
                self.status_bar.showMessage("使用直接权重格式")
            
            # 加载状态字典
            self.status_bar.showMessage("正在应用权重到模型...")
            missing_keys, unexpected_keys = self.sr_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
                
            self.sr_model.to(self.device)
            self.sr_model.eval()
            
            self.model_type = "pytorch"
            self.status_bar.showMessage(f"PyTorch模型加载成功: {method} x{scale}")
            return True
            
        except Exception as e:
            error_msg = f"PyTorch模型加载错误: {str(e)}"
            self.status_bar.showMessage(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False
        
    def run_dnn_super_resolution(self):
        try:
            # 检查图像通道数
            if self.image is None:
                self.status_bar.showMessage("没有图像可处理")
                return None
            
            # 根据模型类型调用不同的推理方法
            if hasattr(self, 'model_type') and self.model_type == "pytorch":
                return self._run_pytorch_inference()
            else:
                return self._run_opencv_inference()
                
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_opencv_inference(self):
        """运行OpenCV DNN模型推理"""
        # 确保图像是3通道的，但保留原始图像类型信息
        input_image = self.image.copy()
        original_is_grayscale = False
        
        if len(input_image.shape) == 2:
            # 单通道灰度图
            original_is_grayscale = True
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        elif len(input_image.shape) == 3 and input_image.shape[2] == 1:
            # 单通道灰度图（带通道维度）
            original_is_grayscale = True
            input_image = cv2.cvtColor(input_image.squeeze(2), cv2.COLOR_GRAY2BGR)
        elif input_image.shape[2] == 4:
            # 将RGBA图像转换为BGR
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)
        
        # 调用SR模型
        result = self.sr_model.upsample(input_image)
        
        # 如果原图是灰度图，将结果转回灰度
        if original_is_grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
        return result
    
    def _run_pytorch_inference(self):
        """运行PyTorch模型推理"""
        try:
            # 预处理图像
            input_image = self._preprocess_image_for_pytorch(self.image)
            
            # 转换为PyTorch张量
            input_tensor = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            input_tensor = input_tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output_tensor = self.sr_model(input_tensor)
            
            # 后处理
            output_array = self._postprocess_pytorch_output(output_tensor)
            
            return output_array
            
        except Exception as e:
            self.status_bar.showMessage(f"PyTorch推理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preprocess_image_for_pytorch(self, image):
        """为PyTorch模型预处理图像"""
        # 复制图像以避免修改原始数据
        input_image = image.copy()
        
        # 转换为灰度图（PyTorch模型期望单通道输入）
        if len(input_image.shape) == 3:
            if input_image.shape[2] == 3:
                # BGR转灰度
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            elif input_image.shape[2] == 4:
                # RGBA转灰度
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2GRAY)
            elif input_image.shape[2] == 1:
                # 已经是单通道，去掉最后一个维度
                input_image = input_image.squeeze(2)
        elif len(input_image.shape) == 2:
            # 已经是灰度图
            pass
        else:
            raise ValueError(f"不支持的图像维度: {input_image.shape}")
        
        # 归一化到[0,1]
        input_image = input_image.astype(np.float32) / 255.0
        
        return input_image
    
    def _postprocess_pytorch_output(self, output_tensor):
        """后处理PyTorch模型输出"""
        # 移除batch和channel维度
        if output_tensor.dim() == 4:  # [1, 1, H, W]
            output_tensor = output_tensor.squeeze(0).squeeze(0)
        elif output_tensor.dim() == 3:  # [1, H, W]
            output_tensor = output_tensor.squeeze(0)
        
        # 转换为numpy数组
        output_array = output_tensor.cpu().numpy()
        
        # 反归一化并裁剪到有效范围
        output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
        
        return output_array
    
    def _cleanup_model(self):
        """清理当前加载的模型"""
        try:
            if self.sr_model is not None:
                if self.model_type == "pytorch":
                    # 清理PyTorch模型
                    if hasattr(self.sr_model, 'cpu'):
                        self.sr_model.cpu()
                    del self.sr_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    # 清理OpenCV模型
                    del self.sr_model
                    
                self.sr_model = None
                self.model_type = None
                
        except Exception as e:
            print(f"模型清理时出错: {str(e)}")

    def save_result(self):
        if self.sr_result is None:
            self.status_bar.showMessage("No result to save!")
            return

        # 获取原始文件名（不包含路径和扩展名）
        if self.image_path:
            default_name = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name += f"_SR_{self.algo_combo.currentText()}_x{self.scale_spinner.value()}"
        else:
            default_name = "super_resolution_result"

        # 打开保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Super-Resolution Result",
            default_name,
            "Images (*.png *.jpg *.bmp);;All Files (*)"
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.sr_result)
                self.status_bar.showMessage(f"Result saved to {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Error saving file: {str(e)}")

    def toggle_freeze(self):
        """冻结/解冻视频流"""
        if self.video_capture is not None:
            self.is_frozen = not self.is_frozen
            if self.is_frozen:
                self.freeze_button.setText("Unfreeze")
                self.timer.stop()
            else:
                self.freeze_button.setText("Freeze")
                if self.play_button.isEnabled() == False:  # 如果处于播放状态
                    self.timer.start(30)
        else:
            self.status_bar.showMessage("没有视频流可以冻结!") 

    def on_item_expanded(self, item):
        """处理目录展开事件，加载子目录内容"""
        # 检查是否已经加载了子项
        if item.childCount() > 0:
            # 检查第一个子项是否是占位符
            first_child = item.child(0)
            if first_child and first_child.text(0) == "Loading...":
                # 移除占位符
                item.removeChild(first_child)
            else:
                # 已经加载过内容，不需要再次加载
                return
            
        # 获取目录路径
        dir_path = item.data(0, Qt.UserRole)
        if os.path.isdir(dir_path):
            # 加载目录内容
            self._add_directory_contents(item, dir_path) 

    # 添加距离测量相关方法
    def enable_distance_measure(self, enable):
        """启用或禁用距离测量功能"""
        self.measure_mode = enable
        
        # 确保有图像查看器
        if not hasattr(self, 'image_viewer') or not self.image_viewer:
            self.status_bar.showMessage("图像查看器未初始化，无法启用测量工具")
            return
            
        # 确保图像查看器有场景
        if not self.image_viewer.scene():
            self.status_bar.showMessage("没有图像可以测量")
            return
        
        # 禁用角度测量
        if enable:
            self.angle_mode = False
        
        if enable:
            # 连接图像查看器鼠标事件
            self.image_viewer.mousePressEvent = self.measure_mouse_press
            self.image_viewer.mouseMoveEvent = self.measure_mouse_move
            self.image_viewer.mouseReleaseEvent = self.measure_mouse_release
            
            # 显示提示信息
            self.status_bar.showMessage("距离测量工具已激活：在图像上点击并拖动以测量距离")
            
            # 重置当前测量线变量，但不清除之前的线
            self.measure_line = None
            self.measure_text = None
            self.measure_start_point = None
            self.measure_end_point = None
        else:
            # 还原图像查看器默认鼠标事件
            # 恢复原始事件处理方法
            if hasattr(self.image_viewer, 'set_tool'):
                # 如果有指针模式，则恢复为指针模式
                if self.pointer_mode:
                    self.image_viewer.set_tool("pointer")
                elif self.hand_mode:
                    self.image_viewer.set_tool("hand")
            else:
                    # 默认恢复到指针模式
                    self.image_viewer.set_tool("pointer")
            
            # 取消当前进行中的测量
            if self.measure_line and self.measure_line.scene():
                self.image_viewer.scene().removeItem(self.measure_line)
                self.measure_line = None
                
            if self.measure_text and self.measure_text.scene():
                self.image_viewer.scene().removeItem(self.measure_text)
                self.measure_text = None
                
            self.measure_start_point = None
            self.measure_end_point = None
            
            self.status_bar.showMessage("距离测量工具已禁用")
    
    def clear_measurement(self):
        """清除所有测量线和文本"""
        scene = self.image_viewer.scene()
        if not scene:
            return
        
        # 清除距离测量
        for line in self.measurement_lines:
            if line and line.scene() == scene:
                scene.removeItem(line)
        
        for text in self.measurement_texts:
            if text and text.scene() == scene:
                scene.removeItem(text)
        
        self.measurement_lines = []
        self.measurement_texts = []
        self.measure_line = None
        self.measure_text = None
        self.measure_start_point = None
        self.measure_end_point = None
        
        # 清除角度测量
        for line in self.angle_lines:
            if line and line.scene() == scene:
                scene.removeItem(line)
        
        for text in self.angle_texts:
            if text and text.scene() == scene:
                scene.removeItem(text)
        
        self.angle_lines = []
        self.angle_texts = []
        self.angle_points = []
        self.angle_current_line = 0
        
        # 强制更新视图
        self.image_viewer.viewport().update()
        
        self.status_bar.showMessage("已清除所有测量")
    
    def measure_mouse_press(self, event):
        """测量工具的鼠标按下事件"""
        if self.measure_mode:
            # 确保图像查看器已经准备好并且有场景
            if not self.image_viewer or not self.image_viewer.scene():
                self.status_bar.showMessage("无法进行测量：图像查看器未准备好")
                return
                
            scene_pos = self.image_viewer.mapToScene(event.pos())
            self.measure_start_point = (scene_pos.x(), scene_pos.y())
            
            # 创建新的测量线，初始长度为0
            self.measure_line = QGraphicsLineItem(
                self.measure_start_point[0],
                self.measure_start_point[1],
                self.measure_start_point[0],
                self.measure_start_point[1]
            )
            
            # 设置线条样式
            pen = QPen(Qt.red)
            pen.setWidth(2)
            self.measure_line.setPen(pen)
            
            # 添加到场景
            self.image_viewer.scene().addItem(self.measure_line)
            
            # 初始化测量文本
            self.measure_text = QGraphicsTextItem("0 px (0.0 mm)")
            self.measure_text.setDefaultTextColor(Qt.red)
            self.measure_text.setPos(self.measure_start_point[0], self.measure_start_point[1] - 20)
            self.image_viewer.scene().addItem(self.measure_text)
            
            # 阻止事件传递
            event.accept()
            return
            # 使用默认的处理方式
        super(type(self.image_viewer), self.image_viewer).mousePressEvent(event)
    
    def measure_mouse_move(self, event):
        """测量工具的鼠标移动事件"""
        if self.measure_mode and self.measure_start_point and self.measure_line:
            # 确保图像查看器和场景存在
            if not self.image_viewer or not self.image_viewer.scene():
                return
                
            scene_pos = self.image_viewer.mapToScene(event.pos())
            self.measure_end_point = (scene_pos.x(), scene_pos.y())
            
            # 更新测量线
            self.measure_line.setLine(
                self.measure_start_point[0],
                self.measure_start_point[1],
                self.measure_end_point[0],
                self.measure_end_point[1]
            )
            
            # 计算像素距离
            pixel_distance = math.sqrt(
                (self.measure_end_point[0] - self.measure_start_point[0]) ** 2 +
                (self.measure_end_point[1] - self.measure_start_point[1]) ** 2
            )
            
            # 转换为物理距离（毫米）
            physical_distance = pixel_distance * self.pixel_scale_mm
            
            # 更新测量文本
            self.measure_text.setPlainText(f"{int(pixel_distance)} px ({physical_distance:.2f} mm)")
            
            # 更新文本位置为线条中点上方
            mid_x = (self.measure_start_point[0] + self.measure_end_point[0]) / 2
            mid_y = (self.measure_start_point[1] + self.measure_end_point[1]) / 2
            self.measure_text.setPos(mid_x - 50, mid_y - 20)
            
            # 强制刷新场景
            self.image_viewer.viewport().update()
            
            # 阻止事件传递
            event.accept()
            return
            # 使用默认的处理方式
        super(type(self.image_viewer), self.image_viewer).mouseMoveEvent(event)
    
    def measure_mouse_release(self, event):
        """测量工具的鼠标释放事件"""
        if self.measure_mode and self.measure_start_point and self.measure_line:
            # 确保图像查看器和场景存在
            if not self.image_viewer or not self.image_viewer.scene():
                return
                
            scene_pos = self.image_viewer.mapToScene(event.pos())
            self.measure_end_point = (scene_pos.x(), scene_pos.y())
            
            # 计算最终像素距离
            pixel_distance = math.sqrt(
                (self.measure_end_point[0] - self.measure_start_point[0]) ** 2 +
                (self.measure_end_point[1] - self.measure_start_point[1]) ** 2
            )
            
            # 转换为物理距离（毫米）
            physical_distance = pixel_distance * self.pixel_scale_mm
            
            # 更新状态栏显示最终测量结果
            self.status_bar.showMessage(f"测量结果: {int(pixel_distance)} 像素 ({physical_distance:.2f} mm)")
            
            # 将当前测量线和文本添加到列表中
            if self.measure_line and self.measure_text:
                self.measurement_lines.append(self.measure_line)
                self.measurement_texts.append(self.measure_text)
                
                # 重置当前测量线和文本，以便下一次测量
                self.measure_line = None
                self.measure_text = None
                self.measure_start_point = None
                self.measure_end_point = None
            
            # 强制刷新场景
            self.image_viewer.viewport().update()
            
            # 阻止事件传递
            event.accept()
            return
            # 使用默认的处理方式
        super(type(self.image_viewer), self.image_viewer).mouseReleaseEvent(event)
    
    # 添加量角器相关方法
    def enable_angle_measure(self, enable):
        """启用或禁用角度测量功能"""
        self.angle_mode = enable
        
        # 确保有图像查看器
        if not hasattr(self, 'image_viewer') or not self.image_viewer:
            self.status_bar.showMessage("图像查看器未初始化，无法启用角度测量工具")
            return
            
        # 确保图像查看器有场景
        if not self.image_viewer.scene():
            self.status_bar.showMessage("没有图像可以测量角度")
            return
        
        # 禁用距离测量
        if enable:
            self.measure_mode = False
        
        if enable:
            # 连接图像查看器鼠标事件
            self.image_viewer.mousePressEvent = self.angle_mouse_press
            self.image_viewer.mouseMoveEvent = self.angle_mouse_move
            self.image_viewer.mouseReleaseEvent = self.angle_mouse_release
            
            # 显示提示信息
            self.status_bar.showMessage("角度测量工具已激活：请依次点击三个点以测量角度(中间点为角的顶点)")
            
            # 重置量角器状态
            self.angle_current_line = 0
            self.angle_points = []
        else:
            # 还原图像查看器默认鼠标事件
            # 恢复原始事件处理方法
            if hasattr(self.image_viewer, 'set_tool'):
                # 如果有指针模式，则恢复为指针模式
                if self.pointer_mode:
                    self.image_viewer.set_tool("pointer")
                elif self.hand_mode:
                    self.image_viewer.set_tool("hand")
            else:
                    # 默认恢复到指针模式
                    self.image_viewer.set_tool("pointer")
            
            # 清理任何正在进行的角度测量
            if self.angle_lines:
                for line in self.angle_lines:
                    if line and line.scene():
                        self.image_viewer.scene().removeItem(line)
                self.angle_lines = []
            
            self.angle_points = []
            self.angle_current_line = 0
            
            self.status_bar.showMessage("角度测量工具已禁用")
    
    def angle_mouse_press(self, event):
        """量角器的鼠标按下事件"""
        if self.angle_mode:
            # 确保图像查看器已经准备好并且有场景
            if not self.image_viewer or not self.image_viewer.scene():
                self.status_bar.showMessage("无法进行角度测量：图像查看器未准备好")
                return
            
            scene_pos = self.image_viewer.mapToScene(event.pos())
            pos = (scene_pos.x(), scene_pos.y())
            
            # 第一次点击，开始第一条线
            if self.angle_current_line == 0:
                self.angle_points = [pos]  # 存储第一个点
                
                # 创建第一条线
                line = QGraphicsLineItem(
                    pos[0], pos[1], pos[0], pos[1]
                )
                pen = QPen(Qt.blue)
                pen.setWidth(2)
                line.setPen(pen)
                self.image_viewer.scene().addItem(line)
                self.angle_lines.append(line)
                
                self.angle_current_line = 1  # 进入第一条线的绘制状态
                
            # 第二次点击，完成第一条线，开始第二条线
            elif self.angle_current_line == 1:
                if len(self.angle_points) == 1:
                    self.angle_points.append(pos)  # 存储第二个点（角的顶点）
                    
                    # 完成第一条线
                    line1 = self.angle_lines[0]
                    line1.setLine(
                        self.angle_points[0][0],
                        self.angle_points[0][1],
                        pos[0],
                        pos[1]
                    )
                    
                    # 创建第二条线
                    line2 = QGraphicsLineItem(
                        pos[0], pos[1], pos[0], pos[1]
                    )
                    pen = QPen(Qt.blue)
                    pen.setWidth(2)
                    line2.setPen(pen)
                    self.image_viewer.scene().addItem(line2)
                    self.angle_lines.append(line2)
                    
                    self.angle_current_line = 2  # 进入第二条线的绘制状态
                    
            # 第三次点击，完成第二条线，计算角度
            elif self.angle_current_line == 2:
                if len(self.angle_points) == 2:
                    self.angle_points.append(pos)  # 存储第三个点
                    
                    # 完成第二条线
                    line2 = self.angle_lines[1]
                    line2.setLine(
                        self.angle_points[1][0],
                        self.angle_points[1][1],
                        pos[0],
                        pos[1]
                    )
                    
                    # 计算角度
                    vertex = self.angle_points[1]
                    point1 = self.angle_points[0]
                    point2 = self.angle_points[2]
                    
                    # 计算向量
                    vector1 = (point1[0] - vertex[0], point1[1] - vertex[1])
                    vector2 = (point2[0] - vertex[0], point2[1] - vertex[1])
                    
                    # 计算向量的模
                    norm1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
                    norm2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
                    
                    # 计算点积
                    dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
                    
                    # 计算角度（弧度）
                    if norm1 * norm2 == 0:
                        angle_rad = 0
                    else:
                        # 使用余弦公式计算角度
                        cos_angle = max(-1, min(1, dot_product / (norm1 * norm2)))
                        angle_rad = math.acos(cos_angle)
                    
                    # 转换为角度
                    angle_deg = math.degrees(angle_rad)
                    
                    # 创建角度文本
                    text = QGraphicsTextItem(f"{angle_deg:.1f}°")
                    text.setDefaultTextColor(Qt.blue)
                    text.setPos(vertex[0] + 10, vertex[1] + 10)
                    self.image_viewer.scene().addItem(text)
                    self.angle_texts.append(text)
                    
                    # 更新状态栏
                    self.status_bar.showMessage(f"测量角度: {angle_deg:.1f}°")
                    
                    # 重置角度测量状态，准备下一次测量
                    self.angle_current_line = 0
                    self.angle_points = []
            
            # 强制更新视图
            self.image_viewer.viewport().update()
            
            # 阻止事件传递
            event.accept()
            return
        
            # 使用默认的处理方式
        super(type(self.image_viewer), self.image_viewer).mousePressEvent(event)
    
    def angle_mouse_move(self, event):
        """量角器的鼠标移动事件"""
        if self.angle_mode:
            # 确保图像查看器已经准备好并且有场景
            if not self.image_viewer or not self.image_viewer.scene():
                return
                
            scene_pos = self.image_viewer.mapToScene(event.pos())
            pos = (scene_pos.x(), scene_pos.y())
            
            # 正在绘制第一条线
            if self.angle_current_line == 1 and len(self.angle_points) == 1:
                # 更新第一条线
                line = self.angle_lines[0]
                line.setLine(
                    self.angle_points[0][0],
                    self.angle_points[0][1],
                    pos[0],
                    pos[1]
                )
                
            # 正在绘制第二条线
            elif self.angle_current_line == 2 and len(self.angle_points) == 2:
                # 更新第二条线
                line = self.angle_lines[1]
                line.setLine(
                    self.angle_points[1][0],
                    self.angle_points[1][1],
                    pos[0],
                    pos[1]
                )
            
            # 强制更新视图
            self.image_viewer.viewport().update()
            
            # 阻止事件传递
            event.accept()
            return
        
            # 使用默认的处理方式
        super(type(self.image_viewer), self.image_viewer).mouseMoveEvent(event)
    
    def angle_mouse_release(self, event):
        """量角器的鼠标释放事件"""
        if self.angle_mode:
            # 确保图像查看器已经准备好并且有场景
            if not self.image_viewer or not self.image_viewer.scene():
                return
                
            # 强制更新视图
            self.image_viewer.viewport().update()
            
            # 阻止事件传递
            event.accept()
            return
        
            # 使用默认的处理方式
        super(type(self.image_viewer), self.image_viewer).mouseReleaseEvent(event)

    # 新增图像处理功能
    def adjust_window_level(self):
        """窗位窗宽调整"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
        
        # 默认窗位和窗宽
        default_level = 128
        default_width = 255
        
        # 如果是灰度图像，计算平均值作为初始窗位
        if len(self.image.shape) == 2 or (len(self.image.shape) == 3 and self.image.shape[2] == 1):
            default_level = int(np.mean(self.image))
        
        # 创建并显示窗位窗宽调整对话框 - 不等待返回结果，对话框内部处理变更
        self.window_level_dialog = WindowLevelDialog(self, default_level, default_width)
        self.window_level_dialog.show()

    def adjust_brightness_contrast(self):
        """亮度对比度调整"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
        
        # 创建并显示亮度对比度调整对话框 - 不等待返回结果，对话框内部处理变更
        self.brightness_contrast_dialog = BrightnessContrastDialog(self, 0, 0)
        self.brightness_contrast_dialog.show()
    
    def apply_sharpen(self):
        """应用锐化滤波"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 复制原图以保留原始数据
            sharpened_img = self.image.copy()
            
            # 定义锐化核
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            
            # 应用卷积
            sharpened_img = cv2.filter2D(sharpened_img, -1, kernel)
            
            # 显示结果
            self.display_image(self.original_image_label, sharpened_img)
            
            # 保存处理后的图像，以便后续操作
            self.image = sharpened_img
            
            self.status_bar.showMessage("锐化滤波已应用")
        except Exception as e:
            self.status_bar.showMessage(f"锐化滤波应用失败: {str(e)}")
    
    def apply_smooth(self):
        """应用平滑滤波"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 复制原图以保留原始数据
            smoothed_img = self.image.copy()
            
            # 应用高斯模糊
            smoothed_img = cv2.GaussianBlur(smoothed_img, (5, 5), 0)
            
            # 显示结果
            self.display_image(self.original_image_label, smoothed_img)
            
            # 保存处理后的图像，以便后续操作
            self.image = smoothed_img
            
            self.status_bar.showMessage("平滑滤波已应用")
        except Exception as e:
            self.status_bar.showMessage(f"平滑滤波应用失败: {str(e)}")
    
    def apply_histogram_eq(self):
        """应用直方图均衡化"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 复制原图以保留原始数据
            equalized_img = self.image.copy()
            
            # 对于彩色图像，在LAB空间中只均衡化亮度通道
            if len(equalized_img.shape) == 3 and equalized_img.shape[2] >= 3:
                # 转换到LAB空间
                lab = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2LAB)
                
                # 分离通道
                l, a, b = cv2.split(lab)
                
                # 对亮度通道进行均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                
                # 合并通道
                limg = cv2.merge((cl, a, b))
                
                # 转换回BGR空间
                equalized_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:
                # 灰度图像直接均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized_img = clahe.apply(equalized_img)
            
            # 显示结果
            self.display_image(self.original_image_label, equalized_img)
            
            # 保存处理后的图像，以便后续操作
            self.image = equalized_img
            
            self.status_bar.showMessage("直方图均衡化已应用")
        except Exception as e:
            self.status_bar.showMessage(f"直方图均衡化应用失败: {str(e)}")
    
    def apply_edge_detection(self):
        """应用边缘检测"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 复制原图以保留原始数据
            img = self.image.copy()
            
            # 如果是彩色图像，先转换为灰度图
            if len(img.shape) == 3 and img.shape[2] >= 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 应用Canny边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 如果原图是彩色的，创建彩色边缘图
            if len(img.shape) == 3 and img.shape[2] >= 3:
                # 创建彩色边缘图像（红色边缘）
                edge_color = np.zeros_like(img)
                edge_color[edges > 0] = [0, 0, 255]  # 红色边缘
                
                # 将边缘叠加到原图
                img = cv2.addWeighted(img, 0.7, edge_color, 0.3, 0)
            else:
                # 灰度图像直接使用边缘图
                img = edges
            
            # 显示结果
            self.display_image(self.original_image_label, img)
            
            # 保存处理后的图像，以便后续操作
            self.image = img
            
            self.status_bar.showMessage("边缘检测已应用")
        except Exception as e:
            self.status_bar.showMessage(f"边缘检测应用失败: {str(e)}")
    
    def apply_flip_horizontal(self):
        """水平翻转图像"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 执行水平翻转
            flipped_img = cv2.flip(self.image, 1)  # 1表示水平翻转
            
            # 显示结果
            self.display_image(self.original_image_label, flipped_img)
            
            # 保存处理后的图像，以便后续操作
            self.image = flipped_img
            
            self.status_bar.showMessage("图像已水平翻转")
        except Exception as e:
            self.status_bar.showMessage(f"水平翻转失败: {str(e)}")
    
    def apply_flip_vertical(self):
        """垂直翻转图像"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 执行垂直翻转
            flipped_img = cv2.flip(self.image, 0)  # 0表示垂直翻转
            
            # 显示结果
            self.display_image(self.original_image_label, flipped_img)
            
            # 保存处理后的图像，以便后续操作
            self.image = flipped_img
            
            self.status_bar.showMessage("图像已垂直翻转")
        except Exception as e:
            self.status_bar.showMessage(f"垂直翻转失败: {str(e)}")
    
    def apply_rotate_90(self):
        """旋转图像90度"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
            
        try:
            # 执行90度旋转
            rotated_img = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            
            # 显示结果
            self.display_image(self.original_image_label, rotated_img)
            
            # 保存处理后的图像，以便后续操作
            self.image = rotated_img
            
            self.status_bar.showMessage("图像已旋转90度")
        except Exception as e:
            self.status_bar.showMessage(f"旋转失败: {str(e)}")
    
    def extract_roi(self):
        """提取感兴趣区域"""
        if self.image is None:
            self.status_bar.showMessage("没有可处理的图像")
            return
        
        # 切换到ROI选择模式
        self.status_bar.showMessage("请在图像上用鼠标拖动选择ROI区域")
        
        # 保存当前图像查看器状态
        self._prev_mouse_press = self.image_viewer.mousePressEvent
        self._prev_mouse_move = self.image_viewer.mouseMoveEvent
        self._prev_mouse_release = self.image_viewer.mouseReleaseEvent
        
        # 创建橡皮筋选择工具
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_viewer)
        self.roi_origin = None
        
        # 设置ROI选择的鼠标事件
        self.image_viewer.mousePressEvent = self.roi_mouse_press
        self.image_viewer.mouseMoveEvent = self.roi_mouse_move
        self.image_viewer.mouseReleaseEvent = self.roi_mouse_release
    
    def roi_mouse_press(self, event):
        """ROI选择的鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 记录起点
            self.roi_origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.roi_origin, QSize()))
            self.rubber_band.show()
            event.accept()
    
    def roi_mouse_move(self, event):
        """ROI选择的鼠标移动事件"""
        if self.roi_origin is not None:
            # 更新橡皮筋矩形
            self.rubber_band.setGeometry(QRect(self.roi_origin, event.pos()).normalized())
            event.accept()
    
    def roi_mouse_release(self, event):
        """ROI选择的鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.roi_origin is not None:
            try:
                # 获取选择区域
                roi_rect = self.rubber_band.geometry()
                
                # 隐藏橡皮筋
                self.rubber_band.hide()
                
                # 转换为场景坐标
                scene_pos1 = self.image_viewer.mapToScene(roi_rect.topLeft())
                scene_pos2 = self.image_viewer.mapToScene(roi_rect.bottomRight())
                
                # 获取ROI区域
                x1, y1 = int(scene_pos1.x()), int(scene_pos1.y())
                x2, y2 = int(scene_pos2.x()), int(scene_pos2.y())
                
                # 确保坐标在图像范围内
                if len(self.image.shape) == 3:
                    h, w, _ = self.image.shape
                else:
                    h, w = self.image.shape
                    
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # 如果选择了有效区域
                if x2 > x1 and y2 > y1:
                    # 提取ROI
                    roi = self.image[y1:y2, x1:x2].copy()  # 使用copy()确保数据连续
                    
                    # 显示ROI信息
                    roi_info = f"ROI形状: {roi.shape}, 类型: {roi.dtype}"
                    print(f"已提取ROI区域: {roi_info}")
                    
                    # 显示ROI
                    self.display_image(self.original_image_label, roi)
                    
                    # 更新图像
                    self.image = roi
                    
                    self.status_bar.showMessage(f"ROI已提取: ({x1},{y1}) to ({x2},{y2}), {roi_info}")
                else:
                    self.status_bar.showMessage("无效的ROI区域")
                
                # 恢复原来的鼠标事件处理
                self.image_viewer.mousePressEvent = self._prev_mouse_press
                self.image_viewer.mouseMoveEvent = self._prev_mouse_move
                self.image_viewer.mouseReleaseEvent = self._prev_mouse_release
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status_bar.showMessage(f"ROI提取失败: {str(e)}")
                
                # 恢复原来的鼠标事件处理
                self.image_viewer.mousePressEvent = self._prev_mouse_press
                self.image_viewer.mouseMoveEvent = self._prev_mouse_move
                self.image_viewer.mouseReleaseEvent = self._prev_mouse_release
            
            event.accept()

# 添加自定义图像查看器类
class ZoomableImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene())
        self.pixmap_item = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)  # 默认为指针模式
        
        # 添加缩放级别跟踪
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
        # 控制是否自动适应视图
        self.should_fit_in_view = True
        
        # 标志，用于跟踪鼠标按下状态和拖动
        self.is_dragging = False
        self.drag_start_pos = None
        
        # 工具模式
        self.current_tool = "pointer"  # 默认为指针工具

    def set_tool(self, tool_name):
        """设置当前工具模式"""
        self.current_tool = tool_name
        
        if tool_name == "pointer":
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.ArrowCursor)
        elif tool_name == "hand":
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.OpenHandCursor)
        
        # 测量工具在SuperResolutionTab中处理

    def wheelEvent(self, event):
        """处理鼠标滚轮事件实现缩放"""
        # 获取滚轮事件信息
        delta = event.angleDelta().y()
        
        # 一旦用户开始缩放，就禁用自动适应视图
        self.should_fit_in_view = False
        
        # 设置缩放因子和方向
        factor = 1.15  # 稍微增加缩放因子，使缩放更明显
        
        # 确保缩放方向直观：向上滚动放大，向下滚动缩小
        if delta > 0:  # 向上滚动
            # 放大
            new_zoom = self.zoom_level * factor
        else:  # 向下滚动
            # 缩小
            new_zoom = self.zoom_level / factor
        
        # 应用缩放限制
        if new_zoom < self.min_zoom:
            new_zoom = self.min_zoom
        elif new_zoom > self.max_zoom:
            new_zoom = self.max_zoom
        
        # 计算实际缩放因子
        actual_factor = new_zoom / self.zoom_level
        
        # 应用缩放，确保缩放量有效
        if abs(actual_factor - 1.0) > 0.001:
            self.scale(actual_factor, actual_factor)
            self.zoom_level = new_zoom
        
        # 阻止事件传递，避免滚动条同时响应
        event.accept()

    def set_image(self, pixmap):
        """设置要显示的图像"""
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        
        # 重置缩放状态
        self.zoom_level = 1.0
        self.should_fit_in_view = True
        
        # 仅在初始设置图像时适应视图
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def fit_in_view(self):
        """调整图像以适应视图大小 (仅在需要时调用)"""
        if self.pixmap_item and self.should_fit_in_view:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.zoom_level = 1.0

    def resizeEvent(self, event):
        """窗口大小改变时的处理"""
        super().resizeEvent(event)
        
        # 只有在初始状态或明确请求时才适应视图
        if self.pixmap_item and self.should_fit_in_view:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MiddleButton:
            # 中键点击重置缩放
            if self.pixmap_item:
                self.should_fit_in_view = True
                self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
                self.zoom_level = 1.0
                event.accept()
                return
                
        elif event.button() == Qt.LeftButton:
            # 左键按下，可能开始拖动
            self.is_dragging = True
            self.drag_start_pos = event.pos()
            if self.current_tool == "hand":
                self.setCursor(Qt.ClosedHandCursor)  # 显示抓取手势
            
        # 调用父类方法
        super().mousePressEvent(event) 
        
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，实现拖动"""
        # 如果处于拖动状态，并且是手型工具模式
        if self.is_dragging and self.current_tool == "hand" and self.drag_start_pos:
            # 实现拖动逻辑
            delta = event.pos() - self.drag_start_pos
            self.drag_start_pos = event.pos()
            
            # 移动视图
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件，结束拖动"""
        if event.button() == Qt.LeftButton:
            # 左键释放，结束拖动
            self.is_dragging = False
            if self.current_tool == "hand":
                self.setCursor(Qt.OpenHandCursor)  # 恢复普通手势
            
        # 调用父类方法
        super().mouseReleaseEvent(event)