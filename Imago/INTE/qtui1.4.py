import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import numpy as np
import cv2
import vtk
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from skimage import transform, io
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QFrame, QSplitter, 
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QSpinBox, QToolBar, QAction,
    QTabWidget, QGroupBox, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsRectItem, QGraphicsItem, QDockWidget, QListWidget, QProgressBar, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush, QColor, QIcon
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


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
        
        
  #Strecth布局      
        # # 添加到主布局
        # self.main_layout.addWidget(self.left_container)
        # self.main_layout.addWidget(self.center_container)
        # self.main_layout.addWidget(self.right_container)
        
        # # 设置各部分的比例
        # self.main_layout.setStretch(0, 2)  # 文件管理器
        # self.main_layout.setStretch(1, 5)  # 视频播放器
        # self.main_layout.setStretch(2, 2)  # SR控制

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
                
                # 开始播放
                self.timer.start(30)  # 30ms per frame
                self.play_button.setEnabled(False)
                self.status_bar.showMessage("已连接到虚拟摄像头")
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
        
        self.play_button.clicked.connect(self.play_video)
        self.replay_button.clicked.connect(self.replay_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.capture_button.clicked.connect(self.capture_frame)
        
        self.video_progress = QProgressBar()
        
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.replay_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.capture_button)
        
        self.center_layout.addWidget(self.video_progress)
        self.center_layout.addLayout(controls_layout)

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
        """递归添加目录内容"""
        try:
            items = os.listdir(parent_path)
            for item in items:
                if item.startswith('.'):  # 跳过隐藏文件
                    continue
                    
                full_path = os.path.join(parent_path, item)
                if os.path.isdir(full_path):
                    dir_item = QTreeWidgetItem(parent_item, [item])
                    dir_item.setData(0, Qt.UserRole, full_path)
                    self._add_directory_contents(dir_item, full_path)
                elif item.lower().endswith(('.png', '.jpg', '.bmp', '.mp4', '.avi')):
                    file_item = QTreeWidgetItem(parent_item, [item])
                    file_item.setData(0, Qt.UserRole, full_path)
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
                image = cv2.imread(file_path)
                if image is not None:
                    # 转换为RGB格式并显示在video_label中
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                    self.image = image
                    self.image_path = file_path
                    self.status_bar.showMessage("Image loaded successfully")
                else:
                    self.status_bar.showMessage("Failed to load image!")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}")

    def create_sr_controls(self):
        """创建超分辨率控制区域"""
        # 图像显示组
        self.image_display_group = QGroupBox("Image Display")
        self.image_display_layout = QHBoxLayout()  # 水平布局

        # 原始图像
        self.original_image_label = QLabel("Original\nImage")
        self.original_image_label.setAlignment(Qt.AlignCenter)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedWidth(10)
        line.setStyleSheet("background-color: (212, 209, 209, 0.273);")

        # 重建图像
        self.reconstructed_image_label = QLabel("Reconstructed\nImage")
        self.reconstructed_image_label.setAlignment(Qt.AlignCenter)

        # 添加到布局
        self.image_display_layout.addWidget(self.original_image_label)
        self.image_display_layout.addWidget(line)
        self.image_display_layout.addWidget(self.reconstructed_image_label)
        
        self.image_display_group.setLayout(self.image_display_layout)
        self.right_layout.addWidget(self.image_display_group)

        font1 = QFont("Microsoft YaHei", 10)
        self.original_image_label.setFont(font1)
        self.reconstructed_image_label.setFont(font1)
        
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
        self.algo_combo.addItems(["Bilinear", "Bicubic", "EDSR", "ESPCN", "FSRCNN", "LAPSRN"])
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
        self.right_layout.addWidget(self.controls_group)

    def open_selected_file(self, item):
        """打开选中的文件"""
        file_path = item.data(0, Qt.UserRole)
        if file_path.lower().endswith(('.mp4', '.avi')):
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
        """更新视频帧"""
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))

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
        """捕获当前帧"""
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                # 保存当前帧
                self.image = frame
                # 显示在original_image_label中
                self.display_image(self.original_image_label, frame)
                self.status_bar.showMessage("已捕获当前帧")
                # 启用超分辨率相关按钮
                self.process_button.setEnabled(True)

    def display_image(self, label, image):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)

    def apply_super_resolution(self):
        if self.image is None:
            self.status_bar.showMessage("No image loaded!")
            return

        scale = self.scale_spinner.value()
        algorithm = self.algo_combo.currentText()
        start_time = time.time()

        if algorithm in ["EDSR", "ESPCN", "FSRCNN", "LAPSRN"]:
            self.load_dnn_sr_model(algorithm, scale)
            sr_image = self.run_dnn_super_resolution()
        elif algorithm in ["Bilinear", "Bicubic"]:
            sr_image = self.run_opencv_interpolation(algorithm, scale)
        else:
            sr_image = None

        end_time = time.time()
        if sr_image is not None:
            self.sr_result = sr_image  # 存储超分辨率结果
            self.display_image(self.reconstructed_image_label, sr_image)
            elapsed_time = end_time - start_time
            self.status_bar.showMessage(f"Processing completed in {elapsed_time:.3f} seconds")
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
        model_paths = {
            "EDSR": f"./model/EDSR_x{scale}.pb",
            "ESPCN": f"./model/ESPCN_x{scale}.pb",
            "FSRCNN": f"./model/FSRCNN_x{scale}.pb",
            "LAPSRN": f"./model/LAPSRN_x{scale}.pb"
        }
        self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr_model.readModel(model_paths[method])
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



# MedSAM模块
class MedSAMTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status_bar = status_bar

        # 配置MedSAM
        self.SAM_MODEL_TYPE = "vit_b"
        self.MedSAM_CKPT_PATH = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
        self.MEDSAM_IMG_INPUT_SIZE = 1024

        # 图形界面参数
        self.half_point_size = 5
        self.point_size = self.half_point_size * 2
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        self.color_idx = 0

        # 状态变量
        self.image_path = None
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None
        self.mask_c = None
        self.img_3c = None

        # 主布局修改为水平布局
        self.main_layout = QHBoxLayout(self)

        # 左侧分割区域
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)

        # 图形视图
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.left_layout.addWidget(self.view)

        # 控制按钮布局
        control_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.save_button = QPushButton("Save Mask")
        self.undo_button = QPushButton("Undo")

        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_mask)
        self.undo_button.clicked.connect(self.undo)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.undo_button)
        self.left_layout.addLayout(control_layout)

        # 右侧EF计算区域
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)

        # EDV图像区域
        self.edv_group = QGroupBox("EDV Image")
        self.edv_layout = QVBoxLayout()
        self.edv_label = QLabel()
        self.edv_layout.addWidget(self.edv_label)
        self.edv_group.setLayout(self.edv_layout)

        # ESV图像区域
        self.esv_group = QGroupBox("ESV Image")
        self.esv_layout = QVBoxLayout()
        self.esv_label = QLabel()
        self.esv_layout.addWidget(self.esv_label)
        self.esv_group.setLayout(self.esv_layout)

        # EF计算控制区
        self.ef_control_group = QGroupBox("EF Calculation")
        self.ef_control_layout = QVBoxLayout()

        self.load_edv_button = QPushButton("Load EDV Mask")
        self.load_esv_button = QPushButton("Load ESV Mask")
        self.calculate_ef_button = QPushButton("Calculate EF")

        self.ef_result_label = QLabel("EF Result: ")
        self.ef_result_label.setAlignment(Qt.AlignCenter)

        self.load_edv_button.clicked.connect(self.load_edv_mask)
        self.load_esv_button.clicked.connect(self.load_esv_mask)
        self.calculate_ef_button.clicked.connect(self.calculate_ef)

        self.ef_control_layout.addWidget(self.load_edv_button)
        self.ef_control_layout.addWidget(self.load_esv_button)
        self.ef_control_layout.addWidget(self.calculate_ef_button)
        self.ef_control_layout.addWidget(self.ef_result_label)

        self.ef_control_group.setLayout(self.ef_control_layout)

        # 添加到右侧布局
        self.right_layout.addWidget(self.edv_group)
        self.right_layout.addWidget(self.esv_group)
        self.right_layout.addWidget(self.ef_control_group)

        # 添加到主布局
        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_widget)

        # EF计算相关变量
        self.edv_mask = None
        self.esv_mask = None

        # 设置设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.status_bar.showMessage("正在加载MedSAM模型...")
        self.load_medsam_model()

        # 连接鼠标事件到视图
        self.view.mousePressEvent = self.mouse_press
        self.view.mouseMoveEvent = self.mouse_move
        self.view.mouseReleaseEvent = self.mouse_release

    def load_medsam_model(self):
        """加载MedSAM模型并计算加载时间"""
        try:
            start_time = time.time()
            
            # 检查模型文件是否存在
            if not os.path.exists(self.MedSAM_CKPT_PATH):
                self.status_bar.showMessage(f"错误: 模型文件未找到，路径: {self.MedSAM_CKPT_PATH}")
                return False
            
            # 加载模型    
            self.status_bar.showMessage("加载MedSAM模型中，请稍候...")
            QApplication.processEvents()  # 更新UI
            
            self.medsam_model = sam_model_registry[self.SAM_MODEL_TYPE](checkpoint=self.MedSAM_CKPT_PATH).to(self.device)
            self.medsam_model.eval()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.status_bar.showMessage(f"MedSAM模型加载完成！耗时: {elapsed_time:.2f}秒")
            print(f"MedSAM Loaded: Cost : {elapsed_time:.2f}秒")
            return True
            
        except Exception as e:
            self.status_bar.showMessage(f"加载MedSAM模型时出错: {str(e)}")
            print(f"加载MedSAM模型时出错: {str(e)}")
            return False

    # 添加推理方法
    @torch.no_grad()
    def medsam_inference(self, img_embed, box_1024, height, width):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.medsam_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    # 添加图像嵌入计算方法
    @torch.no_grad()
    def get_embeddings(self, image):
        print("Calculating embedding...")
        img_1024 = transform.resize(
            image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        embedding = self.medsam_model.image_encoder(img_1024_tensor)
        print("Embedding calculated.")
        return embedding

    # 修改load_image方法
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if not file_path:
            return

        self.image_path = file_path
        self.bg_img = cv2.imread(file_path)
        if self.bg_img is None:
            self.status_bar.showMessage("Failed to load image!")
            return

        # 保存RGB格式的图像
        self.img_3c = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2RGB)

        # 初始化mask
        H, W, _ = self.img_3c.shape
        self.mask_c = np.zeros((H, W, 3), dtype=np.uint8)

        # 计算图像嵌入
        self.embedding = self.get_embeddings(self.bg_img)

        # 显示图像
        self.scene.clear()
        pixmap = self.np2pixmap(self.img_3c)
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(0, 0, W, H)
        self.status_bar.showMessage("Image loaded successfully")

    # 修改mouse_release方法
    def mouse_release(self, event):
        self.is_mouse_down = False
        if self.rect is None:
            return

        scene_pos = self.view.mapToScene(event.pos())
        self.end_pos = (scene_pos.x(), scene_pos.y())

        # 计算边界框
        xmin = min(self.start_pos[0], self.end_pos[0])
        xmax = max(self.start_pos[0], self.end_pos[0])
        ymin = min(self.start_pos[1], self.end_pos[1])
        ymax = max(self.start_pos[1], self.end_pos[1])

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        # 运行MedSAM推理
        sam_mask = self.medsam_inference(self.embedding, box_1024, H, W)

        # 保存当前mask用于撤销
        self.prev_mask = self.mask_c.copy()

        # 更新mask
        self.mask_c[sam_mask != 0] = self.colors[self.color_idx % len(self.colors)]
        self.color_idx += 1

        # 混合显示
        bg = Image.fromarray(self.img_3c)
        mask = Image.fromarray(self.mask_c)
        img = Image.blend(bg, mask, 0.2)

        # 更新显示
        self.scene.clear()
        self.scene.addPixmap(self.np2pixmap(np.array(img)))
        self.status_bar.showMessage("Segmentation completed")

    def save_mask(self):
        """保存分割掩码"""
        if self.mask_c is None:
            self.status_bar.showMessage("No mask to save!")
            return

        save_path = self.image_path.split('.')[0] + '_mask.png' if self.image_path else "mask.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", save_path, "PNG Files (*.png)"
        )

        if file_path:
            # 转换为单通道二值图像
            binary_mask = np.zeros((self.mask_c.shape[0], self.mask_c.shape[1]), dtype=np.uint8)
            binary_mask[np.any(self.mask_c > 0, axis=2)] = 255  # 任何通道有颜色的位置设为255

            try:
                cv2.imwrite(file_path, binary_mask)
                self.status_bar.showMessage(f"Mask saved to {file_path}")
            except Exception as e:
                self.status_bar.showMessage(f"Error saving file: {str(e)}")

    def mouse_press(self, event):
        """鼠标按下事件"""
        scene_pos = self.view.mapToScene(event.pos())
        self.is_mouse_down = True
        self.start_pos = (scene_pos.x(), scene_pos.y())
        self.start_point = QGraphicsEllipseItem(
            scene_pos.x() - self.half_point_size,
            scene_pos.y() - self.half_point_size,
            self.point_size,
            self.point_size
        )
        self.start_point.setBrush(QBrush(QColor(*self.colors[self.color_idx])))
        self.scene.addItem(self.start_point)

    def mouse_move(self, event):
        """鼠标移动事件"""
        if not self.is_mouse_down:
            return

        if self.rect is not None:
            self.scene.removeItem(self.rect)

        scene_pos = self.view.mapToScene(event.pos())
        x = min(scene_pos.x(), self.start_pos[0])
        y = min(scene_pos.y(), self.start_pos[1])
        w = abs(scene_pos.x() - self.start_pos[0])
        h = abs(scene_pos.y() - self.start_pos[1])

        self.rect = QGraphicsRectItem(x, y, w, h)
        self.rect.setPen(QPen(QColor(*self.colors[self.color_idx])))
        self.scene.addItem(self.rect)

    def undo(self):
        if self.prev_mask is None:
            self.status_bar.showMessage("No previous mask record")
            return

        self.color_idx -= 1
        self.mask_c = self.prev_mask.copy()

        # 混合显示
        bg = Image.fromarray(self.img_3c)
        mask = Image.fromarray(self.mask_c)
        img = Image.blend(bg, mask, 0.2)

        # 更新显示
        self.scene.clear()
        self.scene.addPixmap(self.np2pixmap(np.array(img)))
        self.prev_mask = None

    def np2pixmap(self, np_img):
        """转换numpy数组为QPixmap"""
        height, width, channel = np_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg)

    def load_edv_mask(self):
        """加载EDV mask"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDV Mask", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.edv_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.edv_mask is not None:
                self.display_mask(self.edv_label, self.edv_mask)
                self.status_bar.showMessage("EDV mask loaded")

    def load_esv_mask(self):
        """加载ESV mask"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open ESV Mask", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.esv_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.esv_mask is not None:
                self.display_mask(self.esv_label, self.esv_mask)
                self.status_bar.showMessage("ESV mask loaded")

    def display_mask(self, label, mask):
        """显示mask图像"""
        height, width = mask.shape
        bytes_per_line = width
        q_image = QImage(mask.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    def calculate_ef(self):
        """计算EF值"""
        if self.edv_mask is None or self.esv_mask is None:
            self.status_bar.showMessage("Please load both EDV and ESV masks first")
            return

        # 计算非零像素数量作为面积
        edv_area = np.count_nonzero(self.edv_mask)
        esv_area = np.count_nonzero(self.esv_mask)

        # 计算EF
        if edv_area == 0:
            self.status_bar.showMessage("EDV area is zero!")
            return

        ef = (edv_area - esv_area) * 100.0 / edv_area
        self.ef_result_label.setText(f"EF Result: {ef:.2f}%")
        self.status_bar.showMessage(f"EF calculated: {ef:.2f}%")

# VTK重建模块
class VTKReconstructionTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status_bar = status_bar
        self.dicom_directory = None
        self.reader = None
        # 图标设置
        icon_path = "INTE\imris.ico"
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            QApplication.setWindowIcon(app_icon)
            self.setWindowIcon(app_icon)
        
        else:
            self.status_bar.showMessage("Error: Icon file not found!")

        # 主布局
        self.main_layout = QVBoxLayout(self)

        # 控制区
        self.controls_group = QGroupBox("3D Reconstruction Controls")
        self.controls_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_layout)

        self.dcm_button = QPushButton("Load DICOM Directory")
        self.dcm_button.clicked.connect(self.load_dicom)
        self.controls_layout.addWidget(self.dcm_button)

        self.render_combo = QComboBox()
        self.render_combo.addItems(["Volume Rendering", "Surface Rendering"])
        self.controls_layout.addWidget(self.render_combo)

        self.start_button = QPushButton("Start Reconstruction")
        self.start_button.clicked.connect(self.start_reconstruction)
        self.controls_layout.addWidget(self.start_button)

        self.main_layout.addWidget(self.controls_group)

        # VTK显示区
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setAttribute(Qt.WA_AlwaysStackOnTop, False)  # 确保不会始终在顶层
        self.vtk_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # 允许鼠标事件
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.setFocusPolicy(Qt.StrongFocus)  # 设置焦点策略
        self.main_layout.addWidget(self.vtk_widget)

    def load_dicom(self):
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if not directory:
            return

        try:
            # 创建DICOM读取器
            self.reader = vtk.vtkDICOMImageReader()
            self.reader.SetDirectoryName(directory)
            self.reader.Update()
            self.dicom_directory = directory

            self.status_bar.showMessage("DICOM directory loaded successfully")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading DICOM: {str(e)}")
            self.reader = None

    def start_reconstruction(self):
        if not self.reader:
            self.status_bar.showMessage("Please load DICOM data first!")
            return

        render_method = self.render_combo.currentText()

        try:
            # 清除现有渲染
            self.renderer.RemoveAllViewProps()

            if render_method == "Volume Rendering":
                # 创建体绘制管线
                volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
                volume_mapper.SetInputConnection(self.reader.GetOutputPort())

                # 设置颜色传输函数
                volume_color = vtk.vtkColorTransferFunction()
                volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)  # 黑色背景
                volume_color.AddRGBPoint(500, 1.0, 0.5, 0.3)  # 组织
                volume_color.AddRGBPoint(1000, 1.0, 0.9, 0.9)  # 骨骼
                volume_color.AddRGBPoint(1150, 1.0, 1.0, 1.0)  # 高密度区域

                # 设置不透明度传输函数
                volume_scalar_opacity = vtk.vtkPiecewiseFunction()
                volume_scalar_opacity.AddPoint(0, 0.00)  # 完全透明
                volume_scalar_opacity.AddPoint(500, 0.15)  # 半透明
                volume_scalar_opacity.AddPoint(1000, 0.85)  # 较不透明
                volume_scalar_opacity.AddPoint(1150, 1.00)  # 完全不透明

                # 设置体积属性
                volume_property = vtk.vtkVolumeProperty()
                volume_property.SetColor(volume_color)
                volume_property.SetScalarOpacity(volume_scalar_opacity)
                volume_property.ShadeOn()  # 启用阴影
                volume_property.SetInterpolationTypeToLinear()

                # 创建体积
                volume = vtk.vtkVolume()
                volume.SetMapper(volume_mapper)
                volume.SetProperty(volume_property)

                # 添加到渲染器
                self.renderer.AddVolume(volume)

            elif render_method == "Surface Rendering":
                # TODO: 实现表面渲染方法
                pass

            # 重置相机并渲染
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            self.status_bar.showMessage(f"{render_method} completed successfully")

        except Exception as e:
            self.status_bar.showMessage(f"Reconstruction error: {str(e)}")


# 主程序
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 基本窗口设置
        self.setWindowTitle("AI 4 Echocardiography_IMRIS")
        self.setGeometry(100, 100, 1200, 800)

        # 设置全局字体
        font = QFont("Microsoft YaHei", 14)
        QApplication.setFont(font)

        # 状态栏
        self.status_bar = self.statusBar()

        # 创建菜单栏
        self.menubar = self.menuBar()
        self.view_menu = self.menubar.addMenu("View")

        # 创建工具栏
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 创建动作
        self.create_actions()
        
        # 创建标签页
        self.create_tabs()

        # 加载QSS样式
        self.load_style()

    def create_actions(self):
        """创建工具栏动作"""
        self.open_action = QAction("Open", self)
        self.clear_action = QAction("Clear", self)
        self.measure_action = QAction("Measure", self)
        self.tool_action = QAction("Tool", self)
        self.help_action = QAction("Help", self)

        self.open_action.triggered.connect(self.handle_open)
        self.clear_action.triggered.connect(self.handle_clear)

        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.clear_action)
        self.toolbar.addAction(self.measure_action)
        self.toolbar.addAction(self.tool_action)
        self.toolbar.addAction(self.help_action)

    def create_tabs(self):
        """创建标签页"""
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 创建并保存标签页引用
        self.sr_tab = SuperResolutionTab(self, self.status_bar)
        self.dcm_tab = VTKReconstructionTab(self.status_bar)
        self.medsam_tab = MedSAMTab(self.status_bar)

        # 添加标签页
        self.tab_widget.addTab(self.sr_tab, "Super-Resolution")
        self.tab_widget.addTab(self.medsam_tab, "Segmentation")
        self.tab_widget.addTab(self.dcm_tab, "Static 3D Reconstruction")

        # 连接标签页切换信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def load_style(self):
        """加载样式表"""
        try:
            with open("INTE/style.qss", "r") as f:
                style = f.read()
                self.setStyleSheet(style)
        except Exception as e:
            print(f"Error loading style sheet: {str(e)}")

    def on_tab_changed(self, index):
        """处理标签页切换事件"""
        current_tab = self.tab_widget.widget(index)
        self.open_action.setEnabled(isinstance(current_tab, SuperResolutionTab))
        self.clear_action.setEnabled(isinstance(current_tab, SuperResolutionTab))

    def handle_open(self):
        """处理打开文件动作"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if isinstance(current_tab, SuperResolutionTab):
                file_path, _ = QFileDialog.getOpenFileName(
                    self, 
                    "Open File",
                    "",
                    "All Supported Files (*.png *.jpg *.bmp *.mp4 *.avi);;Images (*.png *.jpg *.bmp);;Videos (*.mp4 *.avi)"
                )
                if file_path:
                    if file_path.lower().endswith(('.mp4', '.avi')):
                        current_tab.load_video(file_path)
                    else:
                        current_tab.load_image(file_path)
                    self.status_bar.showMessage(f"Opened: {file_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error opening file: {str(e)}")

    def handle_clear(self):
        """处理清除图像动作"""
        current_tab = self.tab_widget.widget(self.tab_widget.currentIndex())
        if isinstance(current_tab, SuperResolutionTab):
            current_tab.original_image_label.clear()
            current_tab.reconstructed_image_label.clear()
            self.status_bar.showMessage("Images cleared")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
