import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, 
    QTabWidget, QFileDialog, QSplitter, QWidget
)
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QFont, QIcon

# 必要的Qt属性设置 - 在应用程序创建前设置
if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

# 导入自定义模块
from sr import SuperResolutionTab
from seg import MedSAMTab
from recons import VTKReconstructionTab

# 主程序
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 基本窗口设置
        self.setWindowTitle("AI 4 Echocardiography_IMRIS")
        self.setGeometry(100, 100, 1600, 900)
        
        # 确保窗口正常显示（非最小化）
        self.setWindowState(Qt.WindowActive)

        # 预加载资源
        self._load_resources()
        
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
        self.toolbar.setMovable(True)  # 允许移动
        self.toolbar.setFloatable(True)  # 允许浮动
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)  # 指定工具栏区域为顶部
        
        # 添加到视图菜单
        self.view_menu.addAction(self.toolbar.toggleViewAction())

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 创建动作
        self.create_actions()
        
        # 创建标签页
        self.create_tabs()

    def _load_resources(self):
        """预加载和缓存资源"""
        # 加载QSS样式
        self.load_style()
        
        # 设置应用图标
        icon_path = "resource/imris.ico"
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            self.setWindowIcon(app_icon)
            QApplication.setWindowIcon(app_icon)
        else:
            print(f"Warning: Icon file not found at {icon_path}")

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
        # 标签页名称
        self.tab_names = ["Super-Resolution", "Segmentation", "Static 3D Reconstruction"]
        
        # 直接创建所有标签页（不使用懒加载，避免切换问题）
        self.sr_tab = SuperResolutionTab(self, self.status_bar)
        self.tab_widget.addTab(self.sr_tab, self.tab_names[0])
        
        self.seg_tab = MedSAMTab(self.status_bar)
        self.tab_widget.addTab(self.seg_tab, self.tab_names[1])
        
        self.recons_tab = VTKReconstructionTab(self.status_bar)
        self.tab_widget.addTab(self.recons_tab, self.tab_names[2])

        # 连接标签页切换信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def load_style(self):
        """加载样式表"""
        try:
            style_path = "resource/style.qss"
            if os.path.exists(style_path):
                with open(style_path, "r") as f:
                    style = f.read()
                    self.setStyleSheet(style)
            else:
                print(f"Warning: Style file not found at {style_path}")
        except Exception as e:
            print(f"Error loading style sheet: {str(e)}")

    def on_tab_changed(self, index):
        """处理标签页切换事件"""
        # 更新工具栏按钮状态
        current_tab = self.tab_widget.widget(index)
        self.open_action.setEnabled(isinstance(current_tab, SuperResolutionTab))
        self.clear_action.setEnabled(isinstance(current_tab, SuperResolutionTab))
        
        # 针对VTK重建页面的特殊处理
        if index == 2:  # 3D重建标签页
            # 确保VTK渲染器正确初始化
            self.recons_tab.vtk_widget.setVisible(True)
            if hasattr(self.recons_tab.vtk_widget, 'GetRenderWindow'):
                self.recons_tab.vtk_widget.GetRenderWindow().Render()
            
            # 延迟一小段时间以确保正确渲染
            QApplication.processEvents()
        
        # 更新状态栏
        self.status_bar.showMessage(f"Switched to {self.tab_names[index]} tab")

    def handle_open(self):
        """处理打开文件动作"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if isinstance(current_tab, SuperResolutionTab):
                file_path, _ = QFileDialog.getOpenFileName(
                    self, 
                    "Open File",
                    "",
                    "All Supported Files (*.png *.jpg *.bmp *.mp4 *.avi *.mkv);;Images (*.png *.jpg *.bmp);;Videos (*.mp4 *.avi *.mkv)"
                )
                if file_path:
                    # 使用文件扩展名过滤
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in ('.mp4', '.avi', '.mkv'):
                        current_tab.load_video(file_path)
                    elif ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                        current_tab.load_image(file_path)
                    self.status_bar.showMessage(f"Opened: {file_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error opening file: {str(e)}")

    def handle_clear(self):
        """处理清除图像动作"""
        current_tab = self.tab_widget.widget(self.tab_widget.currentIndex())
        if isinstance(current_tab, SuperResolutionTab):
            # 检查是否有图像
            if current_tab.image is not None:
                # 清除原始图像标签
                current_tab.original_image_label.clear()
                
                # 使用image_viewer清除SR结果
                if hasattr(current_tab, 'image_viewer'):
                    current_tab.image_viewer.scene().clear()
                
                # 重置图像变量
                current_tab.image = None
                current_tab.sr_result = None
                # 禁用保存按钮
                current_tab.save_button.setEnabled(False)
                self.status_bar.showMessage("Images cleared")
            else:
                self.status_bar.showMessage("No image to clear")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.setWindowState(Qt.WindowActive)  # 确保窗口正常显示
    main_app.show()
    sys.exit(app.exec_()) 