import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry
from skimage import transform
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QFileDialog, QFrame, QSplitter, 
    QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QGraphicsView, 
    QGraphicsScene, QGraphicsEllipseItem, QGraphicsRectItem, QSlider, QLineEdit, QComboBox,
    QScrollArea, QGridLayout, QFormLayout, QTabWidget
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QPainter
import time


# MedSAM模块
class MedSAMTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status_bar = status_bar

        # 配置MedSAM
        self.SAM_MODEL_TYPE = "vit_b"
        self.MedSAM_CKPT_PATH = "MedSAM\work_dir\MedSAM\medsam_vit_b.pth"
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
        
        # 使用单一垂直布局，恢复为一个标签页
        right_content = QWidget()
        right_content_layout = QVBoxLayout(right_content)
        right_content_layout.setContentsMargins(8, 8, 8, 8)
        right_content_layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("EF Calculation")
        title_label.setStyleSheet("font-size: 12px; font-weight: bold; margin: 0px;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(20)  # 缩小高度
        right_content_layout.addWidget(title_label)
        
        # EDV和ESV显示区域 - 水平布局更紧凑
        masks_display_layout = QHBoxLayout()
        masks_display_layout.setSpacing(5)
        
        # EDV显示区域
        edv_container = QFrame()
        edv_container.setFrameShape(QFrame.StyledPanel)
        edv_container.setFixedHeight(205)  # 调整高度适应200x200的图像
        edv_layout = QVBoxLayout(edv_container)
        edv_layout.setContentsMargins(2, 2, 2, 2)
        edv_layout.setSpacing(0)
        
        self.edv_title = QLabel("EDV")
        self.edv_title.setAlignment(Qt.AlignCenter)
        self.edv_title.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.edv_title.setFixedHeight(15)  # 固定标签高度
        
        self.edv_label = QLabel()
        self.edv_label.setAlignment(Qt.AlignCenter)
        self.edv_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #eeeeee;")
        self.edv_label.setMinimumSize(200, 200)  # 设置为200x200的最小尺寸
        
        edv_layout.addWidget(self.edv_title)
        edv_layout.addWidget(self.edv_label, 1)
        
        # ESV显示区域
        esv_container = QFrame()
        esv_container.setFrameShape(QFrame.StyledPanel)
        esv_container.setFixedHeight(205)  # 调整高度适应200x200的图像
        esv_layout = QVBoxLayout(esv_container)
        esv_layout.setContentsMargins(2, 2, 2, 2)
        esv_layout.setSpacing(0)
        
        self.esv_title = QLabel("ESV")
        self.esv_title.setAlignment(Qt.AlignCenter)
        self.esv_title.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.esv_title.setFixedHeight(15)  # 固定标签高度
        
        self.esv_label = QLabel()
        self.esv_label.setAlignment(Qt.AlignCenter)
        self.esv_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #eeeeee;")
        self.esv_label.setMinimumSize(200, 200)  # 设置为200x200的最小尺寸
        
        esv_layout.addWidget(self.esv_title)
        esv_layout.addWidget(self.esv_label, 1)
        
        # 添加到水平布局
        masks_display_layout.addWidget(edv_container)
        masks_display_layout.addWidget(esv_container)
        
        right_content_layout.addLayout(masks_display_layout)
        
        # 加载按钮部分
        load_buttons_layout = QHBoxLayout()
        
        self.load_edv_button = QPushButton("Load EDV Mask")
        self.load_edv_button.clicked.connect(self.load_edv_mask)
        self.load_edv_button.setFixedHeight(30)
        self.load_edv_button.setMinimumWidth(120)  # 确保按钮宽度足够显示文字
        
        self.load_esv_button = QPushButton("Load ESV Mask")
        self.load_esv_button.clicked.connect(self.load_esv_mask)
        self.load_esv_button.setFixedHeight(30)
        self.load_esv_button.setMinimumWidth(120)  # 确保按钮宽度足够显示文字
        
        load_buttons_layout.addWidget(self.load_edv_button)
        load_buttons_layout.addWidget(self.load_esv_button)
        
        right_content_layout.addLayout(load_buttons_layout)
        
        # Simpson方法参数控件 - 使用FormLayout改进布局
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.StyledPanel)
        params_layout = QFormLayout(params_frame)
        params_layout.setVerticalSpacing(10)
        
        # 添加碟片数量选择器和值显示
        discs_container = QWidget()
        discs_layout = QHBoxLayout(discs_container)
        discs_layout.setContentsMargins(0, 0, 0, 0)
        
        self.disc_count = QSlider(Qt.Horizontal)
        self.disc_count.setRange(10, 50)
        self.disc_count.setValue(20)
        self.disc_count.setTickPosition(QSlider.TicksBelow)
        self.disc_count.setTickInterval(5)
        
        self.disc_value_label = QLabel("20")
        self.disc_value_label.setMinimumWidth(30)
        self.disc_count.valueChanged.connect(lambda v: self.disc_value_label.setText(str(v)))
        
        discs_layout.addWidget(self.disc_count, 4)
        discs_layout.addWidget(self.disc_value_label, 1)
        
        params_layout.addRow("Number of Discs:", discs_container)
        
        # 添加长轴长度校准设置
        self.pixel_scale_input = QLineEdit("0.2")  # 默认值，根据实际情况调整
        params_layout.addRow("Pixel Scale (mm/pixel):", self.pixel_scale_input)
        
        # 添加计算方法选择
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Simpson (Biplane)", "Pixel Count", "Simpson (Single Plane)"])
        params_layout.addRow("Calculation Method:", self.method_combo)
        
        right_content_layout.addWidget(params_frame)
        
        # 添加计算按钮
        self.calculate_ef_button = QPushButton("Calculate EF")
        self.calculate_ef_button.clicked.connect(self.calculate_ef)
        self.calculate_ef_button.setStyleSheet("font-weight: bold; height: 30px;")
        right_content_layout.addWidget(self.calculate_ef_button)
        
        # 添加结果显示 - 紧凑的布局
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_layout = QFormLayout(results_frame)
        results_layout.setVerticalSpacing(8)
        
        # 添加容积显示标签，使用更清晰的格式
        self.edv_volume_label = QLabel("0.00 mL")
        self.edv_volume_label.setStyleSheet("font-weight: bold;")
        results_layout.addRow("EDV Volume:", self.edv_volume_label)
        
        self.esv_volume_label = QLabel("0.00 mL")
        self.esv_volume_label.setStyleSheet("font-weight: bold;")
        results_layout.addRow("ESV Volume:", self.esv_volume_label)
        
        self.ef_result_label = QLabel("0.00%")
        self.ef_result_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        results_layout.addRow("EF:", self.ef_result_label)
        
        self.ef_category_label = QLabel("---")
        self.ef_category_label.setStyleSheet("font-style: italic;")
        results_layout.addRow("Category:", self.ef_category_label)
        
        right_content_layout.addWidget(results_frame)
        
        # 添加提示信息
        status_label = QLabel("Masks loaded: None")
        status_label.setStyleSheet("color: #666666; font-style: italic;")
        status_label.setAlignment(Qt.AlignCenter)
        self.mask_status_label = status_label
        right_content_layout.addWidget(status_label)
        
        # 添加到右侧布局
        self.right_layout.addWidget(right_content)
        
        # 添加到主布局，调整比例更合理
        self.main_layout.addWidget(self.left_widget, 2)
        self.main_layout.addWidget(self.right_widget, 1)

        # EF计算相关变量
        self.edv_mask = None
        self.esv_mask = None

        # 设置设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载模型
        print("Loading MedSAM model...")
        self.medsam_model = sam_model_registry[self.SAM_MODEL_TYPE](checkpoint=self.MedSAM_CKPT_PATH).to(self.device)
        self.medsam_model.eval()
        print("MedSAM model loaded successfully!")

        # 连接鼠标事件到视图
        self.view.mousePressEvent = self.mouse_press
        self.view.mouseMoveEvent = self.mouse_move
        self.view.mouseReleaseEvent = self.mouse_release

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
        """计算图像嵌入并返回嵌入向量和计算时间"""
        self.status_bar.showMessage("正在计算图像嵌入...")
        start_time = time.time()  # 开始计时
        
        img_1024 = transform.resize(
            image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        embedding = self.medsam_model.image_encoder(img_1024_tensor)
        
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算所用时间
        
        print(f"图像嵌入计算用时: {elapsed_time:.2f}秒")
        return embedding, elapsed_time  # 返回embedding和计算时间

    # 修改load_image方法
    def load_image(self):
        """加载图像并显示嵌入计算时间"""
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

        # 计算图像嵌入并记录时间
        embedding_result = self.get_embeddings(self.bg_img)
        self.embedding = embedding_result[0]  # 获取embedding
        elapsed_time = embedding_result[1]  # 获取计算时间

        # 显示图像
        self.scene.clear()
        pixmap = self.np2pixmap(self.img_3c)
        self.scene.addPixmap(pixmap)
        self.view.setSceneRect(0, 0, W, H)
        self.status_bar.showMessage(f"图像加载成功，嵌入计算用时: {elapsed_time:.2f}秒")

    # 修改mouse_release方法
    def mouse_release(self, event):
        """鼠标释放事件，使用已计算的嵌入进行分割"""
        self.is_mouse_down = False
        if self.rect is None or self.embedding is None:
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

        # 记录segmentation推理的开始时间
        seg_start_time = time.time()
        
        # 运行MedSAM推理
        sam_mask = self.medsam_inference(self.embedding, box_1024, H, W)
        
        # 计算segmentation推理时间
        seg_elapsed_time = time.time() - seg_start_time

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
        self.status_bar.showMessage(f"分割完成，推理用时: {seg_elapsed_time:.2f}秒")

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
        if self.img_3c is None:
            return
            
        scene_pos = self.view.mapToScene(event.pos())
        self.is_mouse_down = True
        self.start_pos = (scene_pos.x(), scene_pos.y())
        
        # 创建起始点
        self.start_point = QGraphicsEllipseItem(
            scene_pos.x() - self.half_point_size,
            scene_pos.y() - self.half_point_size,
            self.point_size,
            self.point_size
        )
        self.start_point.setBrush(QBrush(QColor(*self.colors[self.color_idx % len(self.colors)])))
        self.scene.addItem(self.start_point)
        
        # 创建初始矩形
        self.rect = QGraphicsRectItem(
            scene_pos.x(), 
            scene_pos.y(), 
            1, 1  # 初始大小为1x1
        )
        self.rect.setPen(QPen(QColor(*self.colors[self.color_idx % len(self.colors)]), 2))
        self.scene.addItem(self.rect)

    def mouse_move(self, event):
        if not self.is_mouse_down or self.rect is None:
            return

        scene_pos = self.view.mapToScene(event.pos())
        self.end_pos = (scene_pos.x(), scene_pos.y())

        # 更新矩形位置和大小
        x = min(self.start_pos[0], self.end_pos[0])
        y = min(self.start_pos[1], self.end_pos[1])
        width = abs(self.end_pos[0] - self.start_pos[0])
        height = abs(self.end_pos[1] - self.start_pos[1])

        # 安全地更新矩形
        try:
            self.rect.setRect(x, y, width, height)
        except RuntimeError:
            # 如果矩形已被删除，创建新的
            self.rect = QGraphicsRectItem(x, y, width, height)
            self.rect.setPen(QPen(QColor(*self.colors[self.color_idx]), 2))
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

    # 修改load_edv_mask方法，添加状态更新
    def load_edv_mask(self):
        """加载EDV mask"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EDV Mask", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.edv_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.edv_mask is not None:
                self.display_mask(self.edv_label, self.edv_mask)
                self.update_mask_status()
                # 当加载了图像，隐藏标签文字
                self.edv_title.setVisible(False)
                self.status_bar.showMessage("EDV mask loaded")

    # 修改load_esv_mask方法，添加状态更新
    def load_esv_mask(self):
        """加载ESV mask"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open ESV Mask", "", "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.esv_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.esv_mask is not None:
                self.display_mask(self.esv_label, self.esv_mask)
                self.update_mask_status()
                # 当加载了图像，隐藏标签文字
                self.esv_title.setVisible(False)
                self.status_bar.showMessage("ESV mask loaded")
    
    # 添加更新掩码状态的方法
    def update_mask_status(self):
        """更新掩码加载状态"""
        edv_status = "EDV ✓" if self.edv_mask is not None else "EDV ✗"
        esv_status = "ESV ✓" if self.esv_mask is not None else "ESV ✗"
        self.mask_status_label.setText(f"Masks loaded: {edv_status}, {esv_status}")
        if self.edv_mask is not None and self.esv_mask is not None:
            self.mask_status_label.setStyleSheet("color: green; font-style: italic;")
        else:
            self.mask_status_label.setStyleSheet("color: #666666; font-style: italic;")

    def display_mask(self, label, mask):
        """显示mask图像"""
        height, width = mask.shape
        bytes_per_line = width
        q_image = QImage(mask.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        # 使用scaled保持纵横比，显示为200x200
        label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    # 添加计算Simpson双平面法的函数
    def calculate_simpson_biplane(self, mask, pixel_scale_mm=0.2, num_discs=20):
        """
        使用Simpson双平面法计算左心室容积
        
        参数:
        mask - 分割掩码 (二值图像)
        pixel_scale_mm - 像素到实际尺寸的比例 (mm/pixel)
        num_discs - 碟片数量
        
        返回:
        volume_ml - 计算得到的左心室容积 (mL)
        """
        if mask is None:
            return 0
        
        # 找到心室区域的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # 找到最大轮廓（假设这是左心室）
        lv_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(lv_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 获取长轴长度（假设最长的边是长轴）
        width = rect[1][0]
        height = rect[1][1]
        long_axis_length = max(width, height)
        
        # 计算每个碟片的高度
        disc_height = long_axis_length / num_discs
        
        # 获取轮廓的中心线
        M = cv2.moments(lv_contour)
        if M["m00"] == 0:
            return 0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 将中心点旋转到垂直于长轴的方向
        angle = rect[2]
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
        
        # 重新找到旋转后的轮廓
        rotated_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not rotated_contours:
            return 0
        
        rotated_lv_contour = max(rotated_contours, key=cv2.contourArea)
        
        # 计算每个碟片的面积
        total_volume = 0
        
        # 假设长轴和短轴是垂直的
        for i in range(num_discs):
            # 计算当前碟片的上下边界
            y_min = int(cy - long_axis_length/2 + i * disc_height)
            y_max = int(y_min + disc_height)
            
            # 确保边界在图像内
            y_min = max(0, y_min)
            y_max = min(rotated_mask.shape[0]-1, y_max)
            
            # 截取当前碟片区域
            disc_slice = rotated_mask[y_min:y_max, :]
            
            # 计算该碟片区域内的像素数量
            disc_area = np.sum(disc_slice > 0)
            
            # 假设碟片是椭圆形，计算长短轴
            if disc_area > 0:
                # 找到水平方向上的宽度
                disc_width = np.sum(np.any(disc_slice > 0, axis=0))
                
                # 假设这是第一个平面的直径a
                a = disc_width * pixel_scale_mm
                
                # 假设第二个平面的直径b = 0.8a（通常心脏的短轴约为长轴的0.8倍）
                # 这个比例可以根据实际心脏解剖学进行调整
                b = 0.8 * a
                
                # 使用Simpson公式: π/4 * a * b * h
                disc_volume = np.pi/4 * a * b * (disc_height * pixel_scale_mm)
                total_volume += disc_volume
        
        # 返回结果，单位转换为mL (mm³ → mL)
        return total_volume / 1000

    # 添加Simpson单平面法计算
    def calculate_simpson_single_plane(self, mask, pixel_scale_mm=0.2, num_discs=20):
        """使用Simpson单平面法计算左心室容积"""
        if mask is None:
            return 0
        
        # 找到心室区域的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # 找到最大轮廓（假设这是左心室）
        lv_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(lv_contour)
        long_axis_length = max(rect[1][0], rect[1][1])
        
        # 计算每个碟片的高度
        disc_height = long_axis_length / num_discs
        
        # 获取轮廓的中心线
        M = cv2.moments(lv_contour)
        if M["m00"] == 0:
            return 0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 将中心点旋转到垂直于长轴的方向
        angle = rect[2]
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
        
        # 计算每个碟片的体积
        total_volume = 0
        
        for i in range(num_discs):
            # 计算当前碟片的上下边界
            y_min = int(cy - long_axis_length/2 + i * disc_height)
            y_max = int(y_min + disc_height)
            
            # 确保边界在图像内
            y_min = max(0, y_min)
            y_max = min(rotated_mask.shape[0]-1, y_max)
            
            # 截取当前碟片区域
            disc_slice = rotated_mask[y_min:y_max, :]
            
            # 计算该碟片区域内的像素数量
            disc_area = np.sum(disc_slice > 0) * (pixel_scale_mm ** 2)  # 转换为实际面积（mm²）
            
            # 使用单平面Simpson公式: π/4 * (d)² * h
            if disc_area > 0:
                # 假设碟片是圆形
                diameter = 2 * np.sqrt(disc_area / np.pi)
                disc_volume = np.pi/4 * (diameter ** 2) * (disc_height * pixel_scale_mm)
                total_volume += disc_volume
        
        # 返回结果，单位转换为mL (mm³ → mL)
        return total_volume / 1000

    # 修改calculate_ef方法，使用Simpson方法
    def calculate_ef(self):
        """计算射血分数(EF)"""
        if self.edv_mask is None or self.esv_mask is None:
            self.status_bar.showMessage("Please load both EDV and ESV masks first")
            return

        # 获取计算方法
        method = self.method_combo.currentText()
        
        # 获取参数
        try:
            pixel_scale = float(self.pixel_scale_input.text())
            disc_count = self.disc_count.value()
        except ValueError:
            self.status_bar.showMessage("Invalid pixel scale value")
            return
        
        # 根据选择的方法计算体积
        if method == "Simpson (Biplane)":
            edv_vol = self.calculate_simpson_biplane(self.edv_mask, pixel_scale, disc_count)
            esv_vol = self.calculate_simpson_biplane(self.esv_mask, pixel_scale, disc_count)
        elif method == "Simpson (Single Plane)":
            edv_vol = self.calculate_simpson_single_plane(self.edv_mask, pixel_scale, disc_count)
            esv_vol = self.calculate_simpson_single_plane(self.esv_mask, pixel_scale, disc_count)
        else:  # 像素计数法
            # 计算非零像素数量作为面积
            edv_area = np.count_nonzero(self.edv_mask)
            esv_area = np.count_nonzero(self.esv_mask)
            
            # 简单地将面积转换为体积（粗略估计）
            edv_vol = edv_area * pixel_scale * pixel_scale * pixel_scale / 1000  # 转换为mL
            esv_vol = esv_area * pixel_scale * pixel_scale * pixel_scale / 1000  # 转换为mL

        # 计算EF
        if edv_vol <= 0:
            self.status_bar.showMessage("EDV volume is zero or negative!")
            return

        ef = (edv_vol - esv_vol) * 100.0 / edv_vol
        
        # 显示结果
        self.edv_volume_label.setText(f"{edv_vol:.2f} mL")
        self.esv_volume_label.setText(f"{esv_vol:.2f} mL")
        self.ef_result_label.setText(f"{ef:.2f}%")
        
        # 更新EF分类标签
        if ef >= 55:
            category = "Normal EF"
            self.ef_category_label.setStyleSheet("font-style: italic; color: green;")
        elif ef >= 45:
            category = "Mildly Reduced EF"
            self.ef_category_label.setStyleSheet("font-style: italic; color: #66cc00;")
        elif ef >= 30:
            category = "Moderately Reduced EF"
            self.ef_category_label.setStyleSheet("font-style: italic; color: #ffcc00;")
        else:
            category = "Severely Reduced EF"
            self.ef_category_label.setStyleSheet("font-style: italic; color: red;")
            
        self.ef_category_label.setText(category)
        
        # 更新状态栏
        self.status_bar.showMessage(f"EF calculated: {ef:.2f}% using {method}")
        
        # 显示结果的详细解释
        result_message = f"EDV: {edv_vol:.2f} mL, ESV: {esv_vol:.2f} mL, EF: {ef:.2f}% - {category}"
            
        # 更新状态栏显示详细结果
        self.status_bar.showMessage(result_message) 