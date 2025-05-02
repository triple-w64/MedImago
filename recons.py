import os
import vtk
import time  # 添加time模块用于FPS计算
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
    QGroupBox, QComboBox, QFileDialog, QSplitter, QLineEdit, QLabel, QSlider,
    QColorDialog, QFormLayout, QSpinBox, QStyle, QStyleOptionSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QApplication
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# 后台加载线程，避免UI阻塞
class DicomLoaderThread(QThread):
    loading_finished = pyqtSignal(object, str)
    loading_error = pyqtSignal(str)
    
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        
    def run(self):
        try:
            # 创建DICOM读取器
            reader = vtk.vtkDICOMImageReader()
            reader.SetDirectoryName(self.directory)
            reader.Update()
            self.loading_finished.emit(reader, self.directory)
        except Exception as e:
            self.loading_error.emit(str(e))


# VTK重建模块
class VTKReconstructionTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status_bar = status_bar
        self.dicom_directory = None
        self.reader = None
        self.current_model = None  # 保存当前的模型以便保存STL
        
        # FPS计算相关变量
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_text_actor = None
        
        # 主布局改为水平布局
        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)  # 确保正确设置布局
        
        # 创建可拖动的分割器
        self.splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板容器
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)

        # 控制区
        self.controls_group = QGroupBox("3D Reconstruction Controls")
        self.controls_group_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_group_layout)

        self.dcm_button = QPushButton("Load DICOM Directory")
        self.dcm_button.clicked.connect(self.load_dicom)
        self.controls_group_layout.addWidget(self.dcm_button)

        # 渲染方法选择
        render_method_layout = QFormLayout()
        self.render_combo = QComboBox()
        self.render_combo.addItems(["Volume Rendering", "Surface Rendering"])
        self.render_combo.currentIndexChanged.connect(self.update_parameter_visibility)
        render_method_layout.addRow("Rendering Method:", self.render_combo)
        self.controls_group_layout.addLayout(render_method_layout)
        
        # 体绘制参数组
        self.volume_params_group = QGroupBox("Volume Rendering Parameters")
        volume_params_layout = QFormLayout()
        self.volume_params_group.setLayout(volume_params_layout)
        
        # 不透明度调整 - 添加数值显示
        opacity_container = QWidget()
        opacity_layout = QHBoxLayout(opacity_container)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setToolTip("Adjust the opacity of the volume")
        
        self.opacity_value = QLabel("50%")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_value.setText(f"{v}%"))
        
        opacity_layout.addWidget(self.opacity_slider, 4)
        opacity_layout.addWidget(self.opacity_value, 1)
        
        volume_params_layout.addRow("Opacity:", opacity_container)
        
        # 颜色映射选择
        self.color_map_combo = QComboBox()
        self.color_map_combo.addItems(["Default", "Grayscale", "Rainbow", "Hot Metal"])
        volume_params_layout.addRow("Color Mapping:", self.color_map_combo)
        
        # 采样距离 - 添加数值显示
        sampling_container = QWidget()
        sampling_layout = QHBoxLayout(sampling_container)
        sampling_layout.setContentsMargins(0, 0, 0, 0)
        
        self.sampling_slider = QSlider(Qt.Horizontal)
        self.sampling_slider.setRange(1, 20)
        self.sampling_slider.setValue(10)
        self.sampling_slider.setToolTip("Adjust sampling distance (smaller values = higher quality, lower performance)")
        
        self.sampling_value = QLabel("1.0")
        self.sampling_slider.valueChanged.connect(lambda v: self.sampling_value.setText(f"{v/10:.1f}"))
        
        sampling_layout.addWidget(self.sampling_slider, 4)
        sampling_layout.addWidget(self.sampling_value, 1)
        
        volume_params_layout.addRow("Sampling Distance:", sampling_container)
        
        # 添加体绘制参数组
        self.controls_group_layout.addWidget(self.volume_params_group)
        
        # 面绘制参数组
        self.surface_params_group = QGroupBox("Surface Rendering Parameters")
        surface_params_layout = QFormLayout()
        self.surface_params_group.setLayout(surface_params_layout)
        
        # 等值面阈值 - 添加数值显示
        iso_container = QWidget()
        iso_layout = QHBoxLayout(iso_container)
        iso_layout.setContentsMargins(0, 0, 0, 0)
        
        self.iso_value_slider = QSlider(Qt.Horizontal)
        self.iso_value_slider.setRange(100, 1500)
        self.iso_value_slider.setValue(500)
        self.iso_value_slider.setToolTip("Adjust the threshold value for iso-surface extraction")
        
        self.iso_value = QLabel("500")
        self.iso_value_slider.valueChanged.connect(lambda v: self.iso_value.setText(str(v)))
        
        iso_layout.addWidget(self.iso_value_slider, 4)
        iso_layout.addWidget(self.iso_value, 1)
        
        surface_params_layout.addRow("Iso Value:", iso_container)
        
        # 平滑程度 - 添加数值显示
        smooth_container = QWidget()
        smooth_layout = QHBoxLayout(smooth_container)
        smooth_layout.setContentsMargins(0, 0, 0, 0)
        
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 50)
        self.smooth_slider.setValue(20)
        self.smooth_slider.setToolTip("Adjust smoothing iterations")
        
        self.smooth_value = QLabel("20")
        self.smooth_slider.valueChanged.connect(lambda v: self.smooth_value.setText(str(v)))
        
        smooth_layout.addWidget(self.smooth_slider, 4)
        smooth_layout.addWidget(self.smooth_value, 1)
        
        surface_params_layout.addRow("Smoothing:", smooth_container)
        
        # 添加透明度控制
        surface_opacity_container = QWidget()
        surface_opacity_layout = QHBoxLayout(surface_opacity_container)
        surface_opacity_layout.setContentsMargins(0, 0, 0, 0)
        
        self.surface_opacity_slider = QSlider(Qt.Horizontal)
        self.surface_opacity_slider.setRange(1, 100)
        self.surface_opacity_slider.setValue(100)
        self.surface_opacity_slider.setToolTip("Adjust the opacity of the surface")
        
        self.surface_opacity_value = QLabel("100%")
        self.surface_opacity_slider.valueChanged.connect(lambda v: self.surface_opacity_value.setText(f"{v}%"))
        
        surface_opacity_layout.addWidget(self.surface_opacity_slider, 4)
        surface_opacity_layout.addWidget(self.surface_opacity_value, 1)
        
        surface_params_layout.addRow("Surface Opacity:", surface_opacity_container)
        
        # 颜色选择按钮
        self.surface_color = QColor(255, 204, 179)  # 初始肤色
        self.color_button = QPushButton("Select Color")
        self.color_button.clicked.connect(self.select_surface_color)
        self.color_button.setStyleSheet(f"background-color: {self.surface_color.name()}")
        surface_params_layout.addRow("Surface Color:", self.color_button)
        
        # 添加面绘制参数组
        self.controls_group_layout.addWidget(self.surface_params_group)
        
        # 光照设置组
        self.lighting_group = QGroupBox("Lighting Settings")
        lighting_layout = QFormLayout()
        self.lighting_group.setLayout(lighting_layout)
        
        # 环境光强度
        ambient_container = QWidget()
        ambient_layout = QHBoxLayout(ambient_container)
        ambient_layout.setContentsMargins(0, 0, 0, 0)
        
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(30)
        self.ambient_slider.setToolTip("Adjust ambient light intensity")
        
        self.ambient_value = QLabel("0.30")
        self.ambient_slider.valueChanged.connect(lambda v: self.ambient_value.setText(f"{v/100:.2f}"))
        
        ambient_layout.addWidget(self.ambient_slider, 4)
        ambient_layout.addWidget(self.ambient_value, 1)
        
        lighting_layout.addRow("Ambient Light:", ambient_container)
        
        # 漫反射强度
        diffuse_container = QWidget()
        diffuse_layout = QHBoxLayout(diffuse_container)
        diffuse_layout.setContentsMargins(0, 0, 0, 0)
        
        self.diffuse_slider = QSlider(Qt.Horizontal)
        self.diffuse_slider.setRange(0, 100)
        self.diffuse_slider.setValue(70)
        self.diffuse_slider.setToolTip("Adjust diffuse light intensity")
        
        self.diffuse_value = QLabel("0.70")
        self.diffuse_slider.valueChanged.connect(lambda v: self.diffuse_value.setText(f"{v/100:.2f}"))
        
        diffuse_layout.addWidget(self.diffuse_slider, 4)
        diffuse_layout.addWidget(self.diffuse_value, 1)
        
        lighting_layout.addRow("Diffuse Light:", diffuse_container)
        
        # 镜面反射强度
        specular_container = QWidget()
        specular_layout = QHBoxLayout(specular_container)
        specular_layout.setContentsMargins(0, 0, 0, 0)
        
        self.specular_slider = QSlider(Qt.Horizontal)
        self.specular_slider.setRange(0, 100)
        self.specular_slider.setValue(20)
        self.specular_slider.setToolTip("Adjust specular light intensity")
        
        self.specular_value = QLabel("0.20")
        self.specular_slider.valueChanged.connect(lambda v: self.specular_value.setText(f"{v/100:.2f}"))
        
        specular_layout.addWidget(self.specular_slider, 4)
        specular_layout.addWidget(self.specular_value, 1)
        
        lighting_layout.addRow("Specular Light:", specular_container)
        
        # 添加光照设置组
        self.controls_group_layout.addWidget(self.lighting_group)
        
        # 相机设置组
        self.camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()
        self.camera_group.setLayout(camera_layout)
        
        # 仰角
        elevation_container = QWidget()
        elevation_layout = QHBoxLayout(elevation_container)
        elevation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(-90, 90)
        self.elevation_slider.setValue(30)
        self.elevation_slider.setToolTip("Adjust camera elevation angle")
        
        self.elevation_value = QLabel("30°")
        self.elevation_slider.valueChanged.connect(lambda v: self.elevation_value.setText(f"{v}°"))
        
        elevation_layout.addWidget(self.elevation_slider, 4)
        elevation_layout.addWidget(self.elevation_value, 1)
        
        camera_layout.addRow("Elevation:", elevation_container)
        
        # 方位角
        azimuth_container = QWidget()
        azimuth_layout = QHBoxLayout(azimuth_container)
        azimuth_layout.setContentsMargins(0, 0, 0, 0)
        
        self.azimuth_slider = QSlider(Qt.Horizontal)
        self.azimuth_slider.setRange(0, 360)
        self.azimuth_slider.setValue(30)
        self.azimuth_slider.setToolTip("Adjust camera azimuth angle")
        
        self.azimuth_value = QLabel("30°")
        self.azimuth_slider.valueChanged.connect(lambda v: self.azimuth_value.setText(f"{v}°"))
        
        azimuth_layout.addWidget(self.azimuth_slider, 4)
        azimuth_layout.addWidget(self.azimuth_value, 1)
        
        camera_layout.addRow("Azimuth:", azimuth_container)
        
        # 添加相机设置组
        self.controls_group_layout.addWidget(self.camera_group)
        
        # 默认情况下显示体绘制参数
        self.update_parameter_visibility(0)

        self.start_button = QPushButton("Start Reconstruction")
        self.start_button.clicked.connect(self.start_reconstruction)
        self.start_button.setEnabled(False)  # 初始禁用，直到数据加载完成
        self.controls_group_layout.addWidget(self.start_button)
        
        # 保存STL区域
        stl_save_layout = QVBoxLayout()
        stl_file_layout = QHBoxLayout()
        
        self.stl_filename_label = QLabel("STL Filename:")
        self.stl_filename_input = QLineEdit()
        self.stl_filename_input.setPlaceholderText("reconstruction")
        stl_file_layout.addWidget(self.stl_filename_label)
        stl_file_layout.addWidget(self.stl_filename_input)
        
        self.save_stl_button = QPushButton("Save STL File")
        self.save_stl_button.clicked.connect(self.save_stl)
        self.save_stl_button.setEnabled(False)  # 初始禁用，直到重建完成
        
        stl_save_layout.addLayout(stl_file_layout)
        stl_save_layout.addWidget(self.save_stl_button)
        
        self.controls_group_layout.addLayout(stl_save_layout)

        # 添加控制组到左侧布局并添加弹性空间
        self.controls_layout.addWidget(self.controls_group)
        self.controls_layout.addStretch()
        
        # 右侧显示容器
        self.display_container = QWidget()
        self.display_layout = QVBoxLayout(self.display_container)

        # 缓存预设的传输函数以提高性能
        self._cached_transfer_functions = {}

        # VTK显示区
        self.vtk_widget = QVTKRenderWindowInteractor(self.display_container)  # 确保正确的父级
        self.vtk_widget.setAttribute(Qt.WA_AlwaysStackOnTop, False)  # 确保不会始终在顶层
        self.vtk_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # 允许鼠标事件
        self.vtk_widget.setStyleSheet("background-color: #1E1E1E;")  # 设置深色背景
        
        # 创建高性能的渲染器
        self.renderer = vtk.vtkRenderer()
        # 设置增强的光照
        self.renderer.UseFXAAOn()  # 开启抗锯齿
        self.renderer.SetTwoSidedLighting(True)
        self.renderer.SetLightFollowCamera(True)
        
        # 设置渲染窗口
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)
        # 设置背景颜色
        self.renderer.SetBackground(0.12, 0.12, 0.12)  # 设置为RGB #1E1E1E (30, 30, 30)
        
        # 开启性能优化
        render_window.SetMultiSamples(0)  # 禁用多重采样以提高性能
        render_window.SetPointSmoothing(True)
        render_window.SetLineSmoothing(True)
        
        # 设置交互器样式
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.SetInteractorStyle(self.interactor_style)
        
        self.vtk_widget.setFocusPolicy(Qt.StrongFocus)  # 设置焦点策略
        self.display_layout.addWidget(self.vtk_widget)
        
        # 添加到分割器
        self.splitter.addWidget(self.controls_container)
        self.splitter.addWidget(self.display_container)
        
        # 设置分割器的初始大小比例
        self.splitter.setStretchFactor(0, 1)  # 控制区域
        self.splitter.setStretchFactor(1, 4)  # 显示区域
        
        # 添加分割器到主布局并设置为扩展充满
        self.main_layout.addWidget(self.splitter, 1)

        # 创建FPS显示的文本Actor
        self.create_fps_display()
        
        # 创建定时器，用于每秒更新FPS值
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(500)  # 每秒更新一次
        
        # 添加计时回调函数，用于计算每帧的渲染时间和FPS
        self.vtk_widget.GetRenderWindow().AddObserver('RenderEvent', self.count_fps)

    def create_fps_display(self):
        """创建FPS显示的文本Actor"""
        self.fps_text_actor = vtk.vtkTextActor()
        self.fps_text_actor.SetInput("FPS: 0.0")
        self.fps_text_actor.GetTextProperty().SetFontSize(30)
        self.fps_text_actor.GetTextProperty().SetColor(1.0, 1.0, 0.0)  # 黄色
        self.fps_text_actor.GetTextProperty().SetBold(True)
        self.fps_text_actor.GetTextProperty().SetShadow(True)
        
        # 设置文本位置在右上角
        self.fps_text_actor.SetDisplayPosition(10, 10)
        self.fps_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.fps_text_actor.GetPositionCoordinate().SetValue(0.85, 0.95)
        
        # 添加到渲染器
        if hasattr(self, 'renderer'):
            self.renderer.AddActor2D(self.fps_text_actor)

    def count_fps(self, obj, event):
        """计算帧率的回调函数"""
        self.frame_count += 1

    def update_fps(self):
        """更新FPS显示"""
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_time
        
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # 更新FPS文本显示
            if self.fps_text_actor is not None:
                self.fps_text_actor.SetInput(f"FPS: {self.fps:.1f}")
                if hasattr(self, 'vtk_widget') and self.vtk_widget.GetRenderWindow():
                    self.vtk_widget.GetRenderWindow().Render()

    def load_dicom(self):
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if not directory:
            return

        # 禁用按钮，避免重复点击
        self.dcm_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.status_bar.showMessage("Loading DICOM data...")

        # 创建并启动加载线程
        self.loader_thread = DicomLoaderThread(directory)
        self.loader_thread.loading_finished.connect(self.on_dicom_loaded)
        self.loader_thread.loading_error.connect(self.on_dicom_error)
        self.loader_thread.start()

    def on_dicom_loaded(self, reader, directory):
        """DICOM数据加载完成的回调"""
        self.reader = reader
        self.dicom_directory = directory
        self.dcm_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.status_bar.showMessage("DICOM directory loaded successfully")

    def on_dicom_error(self, error_message):
        """DICOM数据加载错误的回调"""
        self.dcm_button.setEnabled(True)
        self.reader = None
        self.status_bar.showMessage(f"Error loading DICOM: {error_message}")

    # 添加颜色选择对话框方法
    def select_surface_color(self):
        color = QColorDialog.getColor(self.surface_color, self, "Select Surface Color")
        if color.isValid():
            self.surface_color = color
            self.color_button.setStyleSheet(f"background-color: {color.name()}")
    
    # 添加参数可见性更新方法
    def update_parameter_visibility(self, index):
        is_volume = index == 0  # Volume Rendering是索引0
        self.volume_params_group.setVisible(is_volume)
        self.surface_params_group.setVisible(not is_volume)

    def _get_cached_transfer_function(self, name, color_map=None, opacity_value=0.5):
        """从缓存获取传输函数，如果不存在或参数变化则创建新的"""
        key = f"{name}_{color_map}_{opacity_value}"
        
        if key not in self._cached_transfer_functions:
            if name == "color":
                # 设置颜色传输函数
                volume_color = vtk.vtkColorTransferFunction()
                
                if color_map == "Grayscale":
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    volume_color.AddRGBPoint(1000, 1.0, 1.0, 1.0)
                elif color_map == "Rainbow":
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    volume_color.AddRGBPoint(400, 0.0, 0.0, 1.0)  # 蓝色
                    volume_color.AddRGBPoint(700, 0.0, 1.0, 0.0)  # 绿色
                    volume_color.AddRGBPoint(1000, 1.0, 0.0, 0.0)  # 红色
                    volume_color.AddRGBPoint(1500, 1.0, 1.0, 1.0)  # 白色
                elif color_map == "Hot Metal":
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    volume_color.AddRGBPoint(400, 0.5, 0.0, 0.0)
                    volume_color.AddRGBPoint(700, 1.0, 0.5, 0.0)
                    volume_color.AddRGBPoint(1000, 1.0, 1.0, 0.0)
                    volume_color.AddRGBPoint(1500, 1.0, 1.0, 1.0)
                else:  # Default
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)    # 黑色背景
                    volume_color.AddRGBPoint(500, 1.0, 0.5, 0.3)  # 组织
                    volume_color.AddRGBPoint(1000, 1.0, 0.9, 0.9) # 骨骼
                    volume_color.AddRGBPoint(1150, 1.0, 1.0, 1.0) # 高密度区域
                
                self._cached_transfer_functions[key] = volume_color
            elif name == "opacity":
                # 设置不透明度传输函数
                volume_scalar_opacity = vtk.vtkPiecewiseFunction()
                # 根据不透明度值调整
                opacity_scale = opacity_value  # 0.0-1.0范围
                volume_scalar_opacity.AddPoint(0, 0.00)
                volume_scalar_opacity.AddPoint(500, 0.15 * opacity_scale)
                volume_scalar_opacity.AddPoint(1000, 0.85 * opacity_scale)
                volume_scalar_opacity.AddPoint(1150, 1.00 * opacity_scale)
                self._cached_transfer_functions[key] = volume_scalar_opacity
                
        return self._cached_transfer_functions[key]

    def start_reconstruction(self):
        if not self.reader:
            self.status_bar.showMessage("Please load DICOM data first!")
            return

        render_method = self.render_combo.currentText()
        
        # 禁用按钮，避免重复点击
        self.start_button.setEnabled(False)
        self.status_bar.showMessage(f"Starting {render_method}...")

        try:
            # 清除现有渲染
            self.renderer.RemoveAllViewProps()
            
            # 重新添加FPS文本Actor
            if self.fps_text_actor:
                self.renderer.AddActor2D(self.fps_text_actor)
                
            self.current_model = None  # 重置当前模型

            if render_method == "Volume Rendering":
                # 获取体绘制参数
                opacity_value = self.opacity_slider.value() / 100.0  # 转换为0-1范围
                color_map = self.color_map_combo.currentText()
                sample_distance = self.sampling_slider.value() / 10.0  # 转换为0.1-2.0范围
                
                # 创建体绘制管线 - 使用更高效的GPU体绘制
                volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
                volume_mapper.SetInputConnection(self.reader.GetOutputPort())
                # 启用自适应采样以提高性能
                volume_mapper.SetAutoAdjustSampleDistances(True)
                volume_mapper.SetSampleDistance(sample_distance)
                volume_mapper.SetBlendModeToComposite()

                # 获取缓存的传输函数
                volume_color = self._get_cached_transfer_function("color", color_map, opacity_value)
                volume_scalar_opacity = self._get_cached_transfer_function("opacity", color_map, opacity_value)

                # 设置体积属性
                volume_property = vtk.vtkVolumeProperty()
                volume_property.SetColor(volume_color)
                volume_property.SetScalarOpacity(volume_scalar_opacity)
                volume_property.ShadeOn()  # 启用阴影
                volume_property.SetInterpolationTypeToLinear()
                volume_property.SetScalarOpacityUnitDistance(0.8)  # 设置透明度单位距离
                
                # 获取并应用光照设置
                ambient = self.ambient_slider.value() / 100.0
                diffuse = self.diffuse_slider.value() / 100.0
                specular = self.specular_slider.value() / 100.0
                
                volume_property.SetAmbient(ambient)
                volume_property.SetDiffuse(diffuse)
                volume_property.SetSpecular(specular)

                # 创建体积
                volume = vtk.vtkVolume()
                volume.SetMapper(volume_mapper)
                volume.SetProperty(volume_property)

                # 添加到渲染器
                self.renderer.AddVolume(volume)
                self.save_stl_button.setEnabled(False)  # 体积渲染不支持STL保存

            elif render_method == "Surface Rendering":
                # 获取面绘制参数
                iso_value = self.iso_value_slider.value()
                smooth_iterations = self.smooth_slider.value()
                surface_color = self.surface_color
                
                # 表面渲染实现
                # 使用MarchingCubes提取等值面
                marchingCubes = vtk.vtkMarchingCubes()
                marchingCubes.SetInputConnection(self.reader.GetOutputPort())
                marchingCubes.SetValue(0, iso_value)  # 使用用户设置的阈值
                
                # 平滑处理
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputConnection(marchingCubes.GetOutputPort())
                smoother.SetNumberOfIterations(smooth_iterations)  # 使用用户设置的平滑度
                smoother.SetPassBand(0.1)
                smoother.SetBoundarySmoothing(False)
                smoother.Update()
                
                # 减少多边形数量以提高性能
                decimation = vtk.vtkDecimatePro()
                decimation.SetInputConnection(smoother.GetOutputPort())
                decimation.SetTargetReduction(0.5)  # 减少50%的多边形
                decimation.PreserveTopologyOn()
                
                # 计算法线
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputConnection(decimation.GetOutputPort())
                normals.SetFeatureAngle(60.0)
                
                # 创建映射器
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())
                mapper.ScalarVisibilityOff()
                
                # 创建actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # 使用用户选择的颜色
                actor.GetProperty().SetColor(
                    surface_color.redF(),
                    surface_color.greenF(),
                    surface_color.blueF()
                )
                
                # 设置透明度
                surface_opacity = self.surface_opacity_slider.value() / 100.0
                if surface_opacity < 1.0:
                    actor.GetProperty().SetOpacity(surface_opacity)
                
                # 设置光照属性
                ambient = self.ambient_slider.value() / 100.0
                diffuse = self.diffuse_slider.value() / 100.0
                specular = self.specular_slider.value() / 100.0
                
                actor.GetProperty().SetAmbient(ambient)
                actor.GetProperty().SetDiffuse(diffuse)
                actor.GetProperty().SetSpecular(specular)
                actor.GetProperty().SetSpecularPower(20)
                
                # 添加到渲染器
                self.renderer.AddActor(actor)
                
                # 保存当前的模型以便保存STL
                self.current_model = normals
                self.save_stl_button.setEnabled(True)  # 启用STL保存按钮

            # 重置相机并渲染
            self.renderer.ResetCamera()
            
            # 修复相机视角设置，避免视图平面法线与上方向向量平行
            camera = self.renderer.GetActiveCamera()
            
            # 设置相机位置，使用球坐标系
            elevation = self.elevation_slider.value()
            azimuth = self.azimuth_slider.value()
            
            # 确保elevation不为90或-90度(这会导致视图平面法线与上方向向量平行)
            if elevation >= 89:
                elevation = 89
            elif elevation <= -89:
                elevation = -89
                
            # 先应用方位角，再应用仰角，避免万向节锁
            camera.SetPosition(0, 0, 0)  # 重置位置
            camera.SetFocalPoint(0, 0, 1)  # 看向z轴正方向
            camera.SetViewUp(0, 1, 0)     # 设置上方向为y轴正方向
            
            # 先旋转方位角
            camera.Azimuth(azimuth)
            # 再旋转仰角
            camera.Elevation(elevation)
            # 不应用滚转角
            camera.SetRoll(0)
            
            # 重置相机到合适的距离
            self.renderer.ResetCamera()
            
            # 设置更好的背景色
            self.renderer.SetBackground(0.1, 0.1, 0.2)  # 深蓝色背景
            
            self.vtk_widget.GetRenderWindow().Render()
            
            # 启动交互器
            if not self.vtk_widget.GetRenderWindow().GetInteractor().GetEnabled():
                self.vtk_widget.GetRenderWindow().GetInteractor().Enable()

            self.status_bar.showMessage(f"{render_method} completed successfully")

        except Exception as e:
            self.status_bar.showMessage(f"Reconstruction error: {str(e)}") 
            self.save_stl_button.setEnabled(False)  # 出错时禁用保存按钮
        finally:
            # 恢复按钮状态
            self.start_button.setEnabled(True)
    
    def save_stl(self):
        """保存当前重建的模型为STL文件"""
        if not self.current_model:
            self.status_bar.showMessage("No model to save! Please perform Surface Rendering first.")
            return
            
        try:
            # 获取用户输入的文件名
            base_filename = self.stl_filename_input.text().strip() or "reconstruction"
            
            # 使用文件对话框获取保存路径
            default_dir = os.path.dirname(self.dicom_directory) if self.dicom_directory else ""
            filepath, _ = QFileDialog.getSaveFileName(
                self, 
                "Save STL File",
                os.path.join(default_dir, f"{base_filename}.stl"),
                "STL Files (*.stl)"
            )
            
            if not filepath:
                return  # 用户取消了保存
                
            # 创建STL写入器
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(filepath)
            stl_writer.SetInputConnection(self.current_model.GetOutputPort())
            stl_writer.Write()
            
            self.status_bar.showMessage(f"STL file saved successfully: {filepath}")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error saving STL file: {str(e)}") 