import os
import vtk
import time  # 添加time模块用于FPS计算
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QPushButton, 
    QGroupBox, QComboBox, QFileDialog, QSplitter, QLineEdit, QLabel, QSlider,
    QColorDialog, QFormLayout, QSpinBox, QStyle, QStyleOptionSlider, QTabWidget, QCheckBox
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
        self.render_combo.addItems(["Volume Rendering", "Surface Rendering", "MPR (Multi-Planar Reconstruction)"])
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
        
        # 添加MPR参数组
        self.mpr_params_group = QGroupBox("MPR Parameters")
        mpr_params_layout = QFormLayout()
        self.mpr_params_group.setLayout(mpr_params_layout)
        
        # MPR平面选择
        self.mpr_plane_combo = QComboBox()
        self.mpr_plane_combo.addItems(["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)", "Oblique"])
        self.mpr_plane_combo.currentIndexChanged.connect(self.update_mpr_plane)
        mpr_params_layout.addRow("Plane:", self.mpr_plane_combo)
        
        # 平面位置滑块
        position_container = QWidget()
        position_layout = QHBoxLayout(position_container)
        position_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mpr_position_slider = QSlider(Qt.Horizontal)
        self.mpr_position_slider.setRange(0, 100)
        self.mpr_position_slider.setValue(50)
        self.mpr_position_slider.valueChanged.connect(self.update_mpr_position)
        
        self.mpr_position_value = QLabel("50%")
        self.mpr_position_slider.valueChanged.connect(lambda v: self.mpr_position_value.setText(f"{v}%"))
        
        position_layout.addWidget(self.mpr_position_slider, 4)
        position_layout.addWidget(self.mpr_position_value, 1)
        
        mpr_params_layout.addRow("Position:", position_container)
        
        # 平面旋转控件(仅用于Oblique模式)
        self.oblique_controls = QWidget()
        oblique_layout = QFormLayout(self.oblique_controls)
        
        # X轴旋转
        x_rotation_container = QWidget()
        x_rotation_layout = QHBoxLayout(x_rotation_container)
        x_rotation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.x_rotation_slider = QSlider(Qt.Horizontal)
        self.x_rotation_slider.setRange(0, 360)
        self.x_rotation_slider.setValue(0)
        self.x_rotation_slider.valueChanged.connect(self.update_oblique_plane)
        
        self.x_rotation_value = QLabel("0°")
        self.x_rotation_slider.valueChanged.connect(lambda v: self.x_rotation_value.setText(f"{v}°"))
        
        x_rotation_layout.addWidget(self.x_rotation_slider, 4)
        x_rotation_layout.addWidget(self.x_rotation_value, 1)
        
        oblique_layout.addRow("X Rotation:", x_rotation_container)
        
        # Y轴旋转
        y_rotation_container = QWidget()
        y_rotation_layout = QHBoxLayout(y_rotation_container)
        y_rotation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.y_rotation_slider = QSlider(Qt.Horizontal)
        self.y_rotation_slider.setRange(0, 360)
        self.y_rotation_slider.setValue(0)
        self.y_rotation_slider.valueChanged.connect(self.update_oblique_plane)
        
        self.y_rotation_value = QLabel("0°")
        self.y_rotation_slider.valueChanged.connect(lambda v: self.y_rotation_value.setText(f"{v}°"))
        
        y_rotation_layout.addWidget(self.y_rotation_slider, 4)
        y_rotation_layout.addWidget(self.y_rotation_value, 1)
        
        oblique_layout.addRow("Y Rotation:", y_rotation_container)
        
        # Z轴旋转
        z_rotation_container = QWidget()
        z_rotation_layout = QHBoxLayout(z_rotation_container)
        z_rotation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.z_rotation_slider = QSlider(Qt.Horizontal)
        self.z_rotation_slider.setRange(0, 360)
        self.z_rotation_slider.setValue(0)
        self.z_rotation_slider.valueChanged.connect(self.update_oblique_plane)
        
        self.z_rotation_value = QLabel("0°")
        self.z_rotation_slider.valueChanged.connect(lambda v: self.z_rotation_value.setText(f"{v}°"))
        
        z_rotation_layout.addWidget(self.z_rotation_slider, 4)
        z_rotation_layout.addWidget(self.z_rotation_value, 1)
        
        oblique_layout.addRow("Z Rotation:", z_rotation_container)
        
        # 初始时隐藏斜截面控件
        self.oblique_controls.setVisible(False)
        mpr_params_layout.addRow("", self.oblique_controls)
        
        # 窗宽窗位调整
        ww_container = QWidget()
        ww_layout = QHBoxLayout(ww_container)
        ww_layout.setContentsMargins(0, 0, 0, 0)
        
        self.window_width_slider = QSlider(Qt.Horizontal)
        self.window_width_slider.setRange(1, 4000)
        self.window_width_slider.setValue(400)
        self.window_width_slider.valueChanged.connect(self.update_window_level)
        
        self.window_width_value = QLabel("400")
        self.window_width_slider.valueChanged.connect(lambda v: self.window_width_value.setText(str(v)))
        
        ww_layout.addWidget(self.window_width_slider, 4)
        ww_layout.addWidget(self.window_width_value, 1)
        
        mpr_params_layout.addRow("Window Width:", ww_container)
        
        # 窗位调整
        wl_container = QWidget()
        wl_layout = QHBoxLayout(wl_container)
        wl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.window_level_slider = QSlider(Qt.Horizontal)
        self.window_level_slider.setRange(-1000, 3000)
        self.window_level_slider.setValue(40)
        self.window_level_slider.valueChanged.connect(self.update_window_level)
        
        self.window_level_value = QLabel("40")
        self.window_level_slider.valueChanged.connect(lambda v: self.window_level_value.setText(str(v)))
        
        wl_layout.addWidget(self.window_level_slider, 4)
        wl_layout.addWidget(self.window_level_value, 1)
        
        mpr_params_layout.addRow("Window Level:", wl_container)
        
        # 多平面视图开关
        self.enable_multiview = QCheckBox("Enable Three-Plane View")
        self.enable_multiview.setChecked(False)
        self.enable_multiview.stateChanged.connect(self.toggle_multiview)
        mpr_params_layout.addRow("", self.enable_multiview)
        
        # 添加MPR参数组到控制布局
        self.controls_group_layout.addWidget(self.mpr_params_group)
        
        # 默认隐藏MPR参数
        self.mpr_params_group.setVisible(False)
        
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

        # 对于MPR多平面视图，创建标签页
        self.view_tabs = QTabWidget()
        self.view_tabs.setVisible(False)  # 初始隐藏标签页
        
        # 创建三个标准平面的显示窗口 - 使用延迟初始化策略
        self.axial_widget = None
        self.coronal_widget = None 
        self.sagittal_widget = None

        # 为每个标准平面创建渲染器
        self.axial_renderer = vtk.vtkRenderer()
        self.axial_renderer.SetBackground(0.2, 0.2, 0.2)

        self.coronal_renderer = vtk.vtkRenderer()
        self.coronal_renderer.SetBackground(0.2, 0.2, 0.2)

        self.sagittal_renderer = vtk.vtkRenderer()
        self.sagittal_renderer.SetBackground(0.2, 0.2, 0.2)

        # 添加占位符标签页 - 实际渲染窗口将在需要时创建
        axial_placeholder = QWidget()
        coronal_placeholder = QWidget() 
        sagittal_placeholder = QWidget()

        # 添加到标签页
        self.view_tabs.addTab(axial_placeholder, "Axial (XY)")
        self.view_tabs.addTab(coronal_placeholder, "Coronal (XZ)")
        self.view_tabs.addTab(sagittal_placeholder, "Sagittal (YZ)")
        
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
        self.display_layout.addWidget(self.view_tabs)
        
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

        # MPR相关变量
        self.mpr_plane = None
        self.mpr_plane_actor = None
        self.mpr_mapper = None
        self.mpr_reslice = None
        self.multiview_enabled = False
        self.mpr_widgets = {
            "Axial (XY)": None,
            "Coronal (XZ)": None,
            "Sagittal (YZ)": None
        }
        self.mpr_renderers = {
            "Axial (XY)": self.axial_renderer,
            "Coronal (XZ)": self.coronal_renderer,
            "Sagittal (YZ)": self.sagittal_renderer
        }
        self.mpr_placeholder_widgets = {
            "Axial (XY)": axial_placeholder,
            "Coronal (XZ)": coronal_placeholder,
            "Sagittal (YZ)": sagittal_placeholder
        }

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
        """更新参数区域的可见性"""
        is_volume = index == 0  # Volume Rendering是索引0
        is_surface = index == 1  # Surface Rendering是索引1
        is_mpr = index == 2     # MPR是索引2
        
        self.volume_params_group.setVisible(is_volume)
        self.surface_params_group.setVisible(is_surface)
        self.mpr_params_group.setVisible(is_mpr)
        
        # 对于MPR模式，切换到对应的界面布局
        if is_mpr and self.reader is not None:
            # 如果启用了多平面视图，显示标签页
            self.view_tabs.setVisible(self.multiview_enabled)
        else:
            self.view_tabs.setVisible(False)
    
    def update_mpr_plane(self, index):
        """更新MPR平面类型"""
        plane_type = self.mpr_plane_combo.currentText()
        # 只有在Oblique模式下才显示旋转控件
        self.oblique_controls.setVisible(plane_type == "Oblique")
        
        # 如果已加载数据，更新MPR显示
        if self.reader is not None and self.mpr_plane is not None:
            self.setup_mpr()
    
    def update_mpr_position(self, value):
        """更新MPR平面位置"""
        if self.reader is None or self.mpr_reslice is None:
            return
            
        # 获取图像尺寸
        extent = self.reader.GetOutput().GetExtent()
        dimensions = [
            extent[1] - extent[0] + 1,
            extent[3] - extent[2] + 1,
            extent[5] - extent[4] + 1
        ]
        
        # 计算相对位置(0-1范围)
        relative_pos = value / 100.0
        
        # 根据当前平面类型设置切片位置
        plane_type = self.mpr_plane_combo.currentText()
        if plane_type == "Axial (XY)":
            position = extent[4] + relative_pos * dimensions[2]
            self.mpr_reslice.SetResliceAxesOrigin(0, 0, position)
        elif plane_type == "Coronal (XZ)":
            position = extent[2] + relative_pos * dimensions[1]
            self.mpr_reslice.SetResliceAxesOrigin(0, position, 0)
        elif plane_type == "Sagittal (YZ)":
            position = extent[0] + relative_pos * dimensions[0]
            self.mpr_reslice.SetResliceAxesOrigin(position, 0, 0)
        else:  # Oblique模式下，不改变原点，而是通过旋转矩阵调整
            # 保持当前原点不变
            pass
        
        # 更新显示
        self.mpr_reslice.Update()
        self.vtk_widget.GetRenderWindow().Render()
        
        # 如果是多视图模式，更新当前活动的视图
        if self.multiview_enabled:
            active_tab = self.view_tabs.currentWidget()
            if active_tab is not None:
                active_tab.GetRenderWindow().Render()
    
    def update_oblique_plane(self):
        """更新斜截面的方向"""
        if self.reader is None or self.mpr_reslice is None or self.mpr_plane_combo.currentText() != "Oblique":
            return
            
        # 获取旋转角度
        x_angle = self.x_rotation_slider.value()
        y_angle = self.y_rotation_slider.value()
        z_angle = self.z_rotation_slider.value()
        
        # 创建旋转矩阵
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.RotateX(x_angle)
        transform.RotateY(y_angle)
        transform.RotateZ(z_angle)
        
        # 应用旋转矩阵
        self.mpr_reslice.SetResliceAxes(transform.GetMatrix())
        
        # 更新显示
        self.mpr_reslice.Update()
        self.vtk_widget.GetRenderWindow().Render()
        
        # 如果是多视图模式，更新当前活动的视图
        if self.multiview_enabled:
            active_tab = self.view_tabs.currentWidget()
            if active_tab is not None:
                active_tab.GetRenderWindow().Render()
    
    def update_window_level(self):
        """更新窗宽窗位"""
        if self.mpr_mapper is None:
            return
            
        window_width = self.window_width_slider.value()
        window_level = self.window_level_slider.value()
        
        # 应用窗宽窗位
        self.mpr_mapper.SetWindow(window_width)
        self.mpr_mapper.SetLevel(window_level)
        
        # 更新显示
        self.vtk_widget.GetRenderWindow().Render()
        
        # 如果是多视图模式，更新所有视图
        if self.multiview_enabled:
            for widget in self.mpr_widgets.values():
                if widget and widget.GetRenderWindow():
                    # 确保渲染器处于选择状态
                    widget.GetRenderWindow().Render()
    
    def toggle_multiview(self, state):
        """切换是否显示三平面视图"""
        self.multiview_enabled = state == Qt.Checked
        
        # 根据当前渲染方法决定是否显示标签页
        is_mpr = self.render_combo.currentText() == "MPR (Multi-Planar Reconstruction)"
        self.view_tabs.setVisible(is_mpr and self.multiview_enabled)
        
        # 如果启用多视图且已加载数据，设置三个平面
        if self.multiview_enabled and self.reader is not None:
            # 延迟到第一次需要时才创建渲染窗口
            self.setup_multiview_mpr()
        else:
            # 确保主视图显示当前选择的平面
            self.setup_mpr()
    
    def init_mpr_view(self, view_name):
        """延迟初始化MPR视图窗口"""
        if self.mpr_widgets[view_name] is not None:
            return  # 已经初始化过了
        
        # 创建新的渲染窗口
        widget = QVTKRenderWindowInteractor()
        
        # 设置渲染器
        render_window = widget.GetRenderWindow()
        render_window.AddRenderer(self.mpr_renderers[view_name])
        
        # 设置交互器样式
        interactor_style = vtk.vtkInteractorStyleImage()
        widget.SetInteractorStyle(interactor_style)
        
        # 存储窗口并替换标签页中的占位符
        self.mpr_widgets[view_name] = widget
        
        # 找到对应标签页的索引并替换占位符
        placeholder = self.mpr_placeholder_widgets[view_name]
        index = self.view_tabs.indexOf(placeholder)
        if index >= 0:
            # 移除旧占位符
            self.view_tabs.removeTab(index)
            # 添加新的渲染窗口
            self.view_tabs.insertTab(index, widget, view_name)
            self.view_tabs.setCurrentIndex(index)

    def setup_multiview_mpr(self):
        """设置三平面MPR视图"""
        if self.reader is None:
            return
        
        # 初始化所有视图窗口
        for view_name in self.mpr_renderers.keys():
            self.init_mpr_view(view_name)
            
        # 为每个平面创建MPR重建
        self.setup_axial_mpr()
        self.setup_coronal_mpr()
        self.setup_sagittal_mpr()
        
        # 渲染所有视图
        for widget in self.mpr_widgets.values():
            if widget and widget.GetRenderWindow():
                # 确保窗口已经初始化
                if not widget.GetRenderWindow().GetNeverRendered():
                    widget.GetRenderWindow().Render()

    def setup_axial_mpr(self):
        """设置轴向MPR视图"""
        # 确保视图已初始化
        view_name = "Axial (XY)"
        self.init_mpr_view(view_name)
        
        renderer = self.mpr_renderers[view_name]
        renderer.RemoveAllViewProps()
        
        # 创建轴向(XY)平面的重切片器
        axial_reslice = vtk.vtkImageReslice()
        axial_reslice.SetInputConnection(self.reader.GetOutputPort())
        axial_reslice.SetOutputDimensionality(2)
        axial_reslice.SetInterpolationModeToLinear()
        
        # 设置轴向平面(XY)
        transform = vtk.vtkTransform()
        transform.Identity()
        axial_reslice.SetResliceAxes(transform.GetMatrix())
        
        # 设置初始位置
        extent = self.reader.GetOutput().GetExtent()
        dimensions = [
            extent[1] - extent[0] + 1,
            extent[3] - extent[2] + 1,
            extent[5] - extent[4] + 1
        ]
        mid_slice_z = extent[4] + dimensions[2] // 2
        axial_reslice.SetResliceAxesOrigin(0, 0, mid_slice_z)
        
        # 创建颜色映射
        axial_mapper = vtk.vtkImageMapToWindowLevelColors()
        axial_mapper.SetInputConnection(axial_reslice.GetOutputPort())
        axial_mapper.SetOutputFormatToRGB()
        
        # 应用窗宽窗位
        axial_mapper.SetWindow(self.window_width_slider.value())
        axial_mapper.SetLevel(self.window_level_slider.value())
        
        # 创建Actor
        axial_actor = vtk.vtkImageActor()
        axial_actor.GetMapper().SetInputConnection(axial_mapper.GetOutputPort())
        
        # 添加到渲染器
        renderer.AddActor(axial_actor)
        renderer.ResetCamera()
        
        # 安全地渲染
        widget = self.mpr_widgets[view_name]
        if widget and widget.GetRenderWindow():
            widget.GetRenderWindow().Render()

    def setup_coronal_mpr(self):
        """设置冠状位MPR视图"""
        # 确保视图已初始化
        view_name = "Coronal (XZ)"
        self.init_mpr_view(view_name)
        
        renderer = self.mpr_renderers[view_name]
        renderer.RemoveAllViewProps()
        
        # 创建冠状位(XZ)平面的重切片器
        coronal_reslice = vtk.vtkImageReslice()
        coronal_reslice.SetInputConnection(self.reader.GetOutputPort())
        coronal_reslice.SetOutputDimensionality(2)
        coronal_reslice.SetInterpolationModeToLinear()
        
        # 设置冠状位平面(XZ)
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.RotateX(90)  # 绕X轴旋转90度，形成XZ平面
        coronal_reslice.SetResliceAxes(transform.GetMatrix())
        
        # 设置初始位置
        extent = self.reader.GetOutput().GetExtent()
        dimensions = [
            extent[1] - extent[0] + 1,
            extent[3] - extent[2] + 1,
            extent[5] - extent[4] + 1
        ]
        mid_slice_y = extent[2] + dimensions[1] // 2
        coronal_reslice.SetResliceAxesOrigin(0, mid_slice_y, 0)
        
        # 创建颜色映射
        coronal_mapper = vtk.vtkImageMapToWindowLevelColors()
        coronal_mapper.SetInputConnection(coronal_reslice.GetOutputPort())
        coronal_mapper.SetOutputFormatToRGB()
        
        # 应用窗宽窗位
        coronal_mapper.SetWindow(self.window_width_slider.value())
        coronal_mapper.SetLevel(self.window_level_slider.value())
        
        # 创建Actor
        coronal_actor = vtk.vtkImageActor()
        coronal_actor.GetMapper().SetInputConnection(coronal_mapper.GetOutputPort())
        
        # 添加到渲染器
        renderer.AddActor(coronal_actor)
        renderer.ResetCamera()
        
        # 安全地渲染
        widget = self.mpr_widgets[view_name]
        if widget and widget.GetRenderWindow():
            widget.GetRenderWindow().Render()

    def setup_sagittal_mpr(self):
        """设置矢状位MPR视图"""
        # 确保视图已初始化
        view_name = "Sagittal (YZ)"
        self.init_mpr_view(view_name)
        
        renderer = self.mpr_renderers[view_name]
        renderer.RemoveAllViewProps()
        
        # 创建矢状位(YZ)平面的重切片器
        sagittal_reslice = vtk.vtkImageReslice()
        sagittal_reslice.SetInputConnection(self.reader.GetOutputPort())
        sagittal_reslice.SetOutputDimensionality(2)
        sagittal_reslice.SetInterpolationModeToLinear()
        
        # 设置矢状位平面(YZ)
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.RotateY(90)  # 绕Y轴旋转90度，形成YZ平面
        sagittal_reslice.SetResliceAxes(transform.GetMatrix())
        
        # 设置初始位置
        extent = self.reader.GetOutput().GetExtent()
        dimensions = [
            extent[1] - extent[0] + 1,
            extent[3] - extent[2] + 1,
            extent[5] - extent[4] + 1
        ]
        mid_slice_x = extent[0] + dimensions[0] // 2
        sagittal_reslice.SetResliceAxesOrigin(mid_slice_x, 0, 0)
        
        # 创建颜色映射
        sagittal_mapper = vtk.vtkImageMapToWindowLevelColors()
        sagittal_mapper.SetInputConnection(sagittal_reslice.GetOutputPort())
        sagittal_mapper.SetOutputFormatToRGB()
        
        # 应用窗宽窗位
        sagittal_mapper.SetWindow(self.window_width_slider.value())
        sagittal_mapper.SetLevel(self.window_level_slider.value())
        
        # 创建Actor
        sagittal_actor = vtk.vtkImageActor()
        sagittal_actor.GetMapper().SetInputConnection(sagittal_mapper.GetOutputPort())
        
        # 添加到渲染器
        renderer.AddActor(sagittal_actor)
        renderer.ResetCamera()
        
        # 安全地渲染
        widget = self.mpr_widgets[view_name]
        if widget and widget.GetRenderWindow():
            widget.GetRenderWindow().Render()

    def setup_mpr(self):
        """设置单一MPR视图"""
        if self.reader is None:
            return
            
        # 清除现有渲染
        self.renderer.RemoveAllViewProps()
        
        # 重新添加FPS文本Actor
        if self.fps_text_actor:
            self.renderer.AddActor2D(self.fps_text_actor)
        
        # 创建重切片器
        self.mpr_reslice = vtk.vtkImageReslice()
        self.mpr_reslice.SetInputConnection(self.reader.GetOutputPort())
        self.mpr_reslice.SetOutputDimensionality(2)
        self.mpr_reslice.SetInterpolationModeToLinear()
        
        # 获取图像尺寸
        extent = self.reader.GetOutput().GetExtent()
        dimensions = [
            extent[1] - extent[0] + 1,
            extent[3] - extent[2] + 1,
            extent[5] - extent[4] + 1
        ]
        
        # 根据平面类型设置方向和位置
        plane_type = self.mpr_plane_combo.currentText()
        if plane_type == "Axial (XY)":
            # 轴向平面(XY)
            transform = vtk.vtkTransform()
            transform.Identity()
            self.mpr_reslice.SetResliceAxes(transform.GetMatrix())
            
            # 设置到中间位置
            mid_slice_z = extent[4] + dimensions[2] // 2
            self.mpr_reslice.SetResliceAxesOrigin(0, 0, mid_slice_z)
            
        elif plane_type == "Coronal (XZ)":
            # 冠状位平面(XZ)
            transform = vtk.vtkTransform()
            transform.Identity()
            transform.RotateX(90)  # 绕X轴旋转90度，形成XZ平面
            self.mpr_reslice.SetResliceAxes(transform.GetMatrix())
            
            # 设置到中间位置
            mid_slice_y = extent[2] + dimensions[1] // 2
            self.mpr_reslice.SetResliceAxesOrigin(0, mid_slice_y, 0)
            
        elif plane_type == "Sagittal (YZ)":
            # 矢状位平面(YZ)
            transform = vtk.vtkTransform()
            transform.Identity()
            transform.RotateY(90)  # 绕Y轴旋转90度，形成YZ平面
            self.mpr_reslice.SetResliceAxes(transform.GetMatrix())
            
            # 设置到中间位置
            mid_slice_x = extent[0] + dimensions[0] // 2
            self.mpr_reslice.SetResliceAxesOrigin(mid_slice_x, 0, 0)
            
        else:  # Oblique
            # 斜截面
            transform = vtk.vtkTransform()
            transform.Identity()
            transform.RotateX(self.x_rotation_slider.value())
            transform.RotateY(self.y_rotation_slider.value())
            transform.RotateZ(self.z_rotation_slider.value())
            self.mpr_reslice.SetResliceAxes(transform.GetMatrix())
            
            # 设置到中心位置
            center = [
                (extent[0] + extent[1]) / 2,
                (extent[2] + extent[3]) / 2,
                (extent[4] + extent[5]) / 2
            ]
            self.mpr_reslice.SetResliceAxesOrigin(center)
        
        # 创建颜色映射
        self.mpr_mapper = vtk.vtkImageMapToWindowLevelColors()
        self.mpr_mapper.SetInputConnection(self.mpr_reslice.GetOutputPort())
        self.mpr_mapper.SetOutputFormatToRGB()
        
        # 应用窗宽窗位
        self.mpr_mapper.SetWindow(self.window_width_slider.value())
        self.mpr_mapper.SetLevel(self.window_level_slider.value())
        
        # 创建Actor
        self.mpr_plane_actor = vtk.vtkImageActor()
        self.mpr_plane_actor.GetMapper().SetInputConnection(self.mpr_mapper.GetOutputPort())
        
        # 添加到渲染器
        self.renderer.AddActor(self.mpr_plane_actor)
        
        # 重置相机
        self.renderer.ResetCamera()
        
        # 更新显示
        self.vtk_widget.GetRenderWindow().Render()
        
        # 更新位置滑块的取值范围
        self.mpr_position_slider.setValue(50)  # 重置到中间位置
    
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
                self.view_tabs.setVisible(False)  # 确保标签页不可见

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
                self.view_tabs.setVisible(False)  # 确保标签页不可见

            elif render_method == "MPR (Multi-Planar Reconstruction)":
                # 设置MPR显示
                if self.multiview_enabled:
                    # 多平面视图
                    self.setup_multiview_mpr()
                    self.view_tabs.setVisible(True)
                else:
                    # 单平面视图
                    self.setup_mpr()
                    self.view_tabs.setVisible(False)
                
                self.save_stl_button.setEnabled(False)  # MPR不支持STL保存

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