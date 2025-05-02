import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QFileDialog, QFrame, QSplitter, 
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont


# 超分辨率模块
class SuperResolutionTab(QWidget):
    def __init__(self, parent, status_bar):
        super().__init__(parent)
        self.main_window = parent
        self.status_bar = status_bar
        
        # 添加视频相关的初始化
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.image = None
        self.image_path = None
        self.sr_result = None
        
        # 主布局改为水平布局
        self.main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        # 创建左中右三个部分的容器
        self.left_container = QWidget()
        self.center_container = QWidget()
        self.right_container = QWidget()
        
        # 设置布局
        self.left_layout = QVBoxLayout(self.left_container)
        self.center_layout = QVBoxLayout(self.center_container)
        self.right_layout = QVBoxLayout(self.right_container)
        
        # 创建组件
        self.create_file_manager()
        self.create_video_player()
        self.create_sr_controls()
        
        # 将三个容器添加到 QSplitter
        splitter.addWidget(self.left_container)
        splitter.addWidget(self.center_container)
        splitter.addWidget(self.right_container)
        
        # 设置拉伸因子(可自由调整)
        splitter.setStretchFactor(0, 2)  # 文件管理器
        splitter.setStretchFactor(1, 5)  # 视频播放器
        splitter.setStretchFactor(2, 2)  # SR控制

        # 添加到主布局
        self.main_layout.addWidget(splitter)

    def create_file_manager(self):
        """创建文件管理器"""
        # 使用QTreeWidget
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("Files")
        self.file_tree.itemDoubleClicked.connect(self.open_selected_file)
        
        # 创建水平布局放置按钮
        button_layout = QHBoxLayout()
        
        load_folder_btn = QPushButton("Load Folder")
        load_folder_btn.clicked.connect(self.load_folder)
        
        connect_stream_btn = QPushButton("Connect Streaming") 
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
        
        self.play_button = QPushButton("Play")
        self.replay_button = QPushButton("Replay")
        self.stop_button = QPushButton("Stop")
        self.capture_button = QPushButton("Capture Frame")
        self.freeze_button = QPushButton("Freeze")
        
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
                # 延迟加载子目录，只有当用户展开目录时才加载
                # 这里可以添加一个展开监听器
            
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
        if len(image.shape) == 2 or image.shape[2] == 1:  # 灰度图像
            h, w = image.shape if len(image.shape) == 2 else image.shape[:2]
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # 彩色图像
            if image.shape[2] == 4:  # RGBA图像
                # 转换为RGB格式
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:  # BGR图像
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def display_image(self, label, image):
        """显示图像到标签"""
        if image is not None:
            self._display_image_in_label(image, label)

    def create_sr_controls(self):
        """创建超分辨率控制区域"""
        # 使用QSplitter使控件可以拉伸调整大小
        self.right_splitter = QSplitter(Qt.Vertical)
        
        # 图像显示组
        self.image_display_group = QGroupBox("Image Display")
        self.image_display_layout = QVBoxLayout()  # 改为垂直布局

        # 原始图像（同时也用于显示重建结果）
        self.original_image_label = QLabel("Original/Reconstructed Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)

        # 添加到布局
        self.image_display_layout.addWidget(self.original_image_label)
        
        self.image_display_group.setLayout(self.image_display_layout)
        
        font1 = QFont("Microsoft YaHei", 10)
        self.original_image_label.setFont(font1)
        
        # 控制组
        self.controls_group = QGroupBox("SR Controls")
        self.controls_layout = QVBoxLayout()

        # Scale Factor选择（水平布局）
        scale_layout = QHBoxLayout()
        self.scale_label = QLabel("Scale Factor:")
        self.scale_spinner = QSpinBox()
        self.scale_spinner.setRange(2, 8)
        self.scale_spinner.setValue(2)
        scale_layout.addWidget(self.scale_label)
        scale_layout.addWidget(self.scale_spinner)
        scale_layout.addStretch()
        
        # Algorithm选择（水平布局）
        algo_layout = QHBoxLayout()
        self.algo_label = QLabel("SR Algorithm:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Bilinear", "Bicubic", "EDSR", "ESPCN", "FSRCNN", "FFSCN", "LAPSRN"])
        algo_layout.addWidget(self.algo_label)
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addStretch()
        
        self.controls_layout.addLayout(scale_layout)
        self.controls_layout.addLayout(algo_layout)
        
        self.process_button = QPushButton("Apply SR")
        self.process_button.clicked.connect(self.apply_super_resolution)
        
        self.save_button = QPushButton("Save Result")
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
        start_time = time.time()

        if algorithm in ["EDSR", "ESPCN", "FSRCNN", "FFSCN", "LAPSRN"]:
            self.load_dnn_sr_model(algorithm, scale)
            sr_image = self.run_dnn_super_resolution()
        elif algorithm in ["Bilinear", "Bicubic"]:
            sr_image = self.run_opencv_interpolation(algorithm, scale)
        else:
            sr_image = None

        end_time = time.time()
        if sr_image is not None:
            self.sr_result = sr_image  # 存储超分辨率结果
            # 在原始图像标签上显示重建结果
            self.display_image(self.original_image_label, sr_image)
            elapsed_time = end_time - start_time
            self.status_bar.showMessage(f"Processing completed in {elapsed_time:.4f} seconds")
            self.save_button.setEnabled(True)  # 启用保存按钮
        else:
            self.status_bar.showMessage("Super-resolution failed!")
            self.save_button.setEnabled(False)

    def run_opencv_interpolation(self, method, scale):
        methods = {
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC
        }
        return cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=methods[method])

    def load_dnn_sr_model(self, method, scale):
        """优化模型加载过程"""
        model_paths = {
            "EDSR": f"./Platform/Model/EDSR_x{scale}.pb",
            "ESPCN": f"./Platform/Model/ESPCN_x{scale}.pb",
            "FSRCNN": f"./Platform/Model/FSRCNN_x{scale}.pb",
            "SCN": f"./Platform/Model/FFSCN_x{scale}.pb",
            "LAPSRN": f"./Platform/Model/LAPSRN_x{scale}.pb"
        }
        
        model_path = model_paths[method]
        
        # 检查模型是否已经加载过，避免重复加载相同的模型
        if hasattr(self, '_current_model') and self._current_model == (method, scale):
            return
            
        # 记录当前加载的模型
        self._current_model = (method, scale)
            
        # 加载模型
        self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr_model.readModel(model_path)
        self.sr_model.setModel(method.lower(), scale)
        
    def run_dnn_super_resolution(self):
        try:
            return self.sr_model.upsample(self.image)
        except Exception as e:
            self.status_bar.showMessage(f"Error: {e}")
            return None

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