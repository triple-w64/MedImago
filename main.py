import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, 
    QTabWidget, QFileDialog, QSplitter, QWidget, QDesktopWidget,
    QTextBrowser, QVBoxLayout, QDialog, QMenu, QToolButton
)
from PyQt5.QtCore import Qt, QCoreApplication, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QTextCursor

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

# 帮助对话框类
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MedImago Help")
        self.resize(800, 600)  # 设置合适的大小
        
        layout = QVBoxLayout(self)
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)  # 允许打开外部链接
        layout.addWidget(self.text_browser)
        
        # 加载README内容
        self.load_readme()
        
    def load_readme(self):
        """加载README.md文件内容"""
        readme_path = os.path.join("README.md")  # 修改为加载项目根目录的README
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # 简单地将markdown内容转换为HTML
                    html_content = self.markdown_to_html(content)
                    self.text_browser.setHtml(html_content)
                    # 滚动到顶部
                    cursor = self.text_browser.textCursor()
                    cursor.setPosition(0)
                    self.text_browser.setTextCursor(cursor)
            except Exception as e:
                self.text_browser.setText(f"Error loading README file: {str(e)}")
        else:
            self.text_browser.setText("README file not found.")
    
    def markdown_to_html(self, markdown_text):
        """简单的Markdown到HTML转换"""
        html = "<html><body style='font-family: Arial; line-height: 1.4;'>"
        
        # 处理标题
        markdown_text = markdown_text.replace("# ", "<h1>").replace(" #", "</h1>")
        markdown_text = markdown_text.replace("## ", "<h2>").replace(" ##", "</h2>")
        markdown_text = markdown_text.replace("### ", "<h3>").replace(" ###", "</h3>")
        
        # 处理链接 [text](url)
        import re
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        markdown_text = re.sub(link_pattern, r'<a href="\2">\1</a>', markdown_text)
        
        # 处理代码块
        markdown_text = markdown_text.replace("```bash", "<pre><code>")
        markdown_text = markdown_text.replace("```", "</code></pre>")
        
        # 替换换行符
        markdown_text = markdown_text.replace("\n", "<br>")
        
        html += markdown_text + "</body></html>"
        return html

# 主程序
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 基本窗口设置
        self.setWindowTitle("AI 4 Echocardiography_IMRIS")
        
        # 设置更合理的初始窗口大小
        screen_size = QDesktopWidget().availableGeometry().size()
        window_width = min(1600, screen_size.width() - 100)
        window_height = min(900, screen_size.height() - 100)
        self.resize(window_width, window_height)
        
        # 将窗口居中显示
        self.center_window()

        # 预加载资源
        self._load_resources()
        
        # 设置全局字体
        font = QFont("Microsoft YaHei", 14)
        QApplication.setFont(font)

        # 状态栏
        self.status_bar = self.statusBar()

        # 创建菜单栏
        self.menubar = self.menuBar()
        self.view_menu = self.menubar.addMenu("视图")

        # 创建工具栏
        self.toolbar = QToolBar("主工具栏")
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
        
        # 设置应用程序默认为全屏显示
        # 直接设置为最大化
        self.showMaximized()
    
    def center_window(self):
        """将窗口居中显示"""
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(max(0, x), max(0, y))

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
        # 创建动作并添加图标
        self.open_action = QAction("打开", self)
        self.open_action.setToolTip("打开文件")
        
        self.clear_action = QAction("清除", self)
        self.clear_action.setToolTip("清除当前内容")
        
        # 工具菜单相关
        self.tool_action = QAction("工具", self)
        self.tool_action.setToolTip("工具选项")
        
        # 创建工具菜单
        self.tool_menu = QMenu(self)
        
        # 距离测量工具移到工具菜单中
        self.distance_measure_action = QAction("距离测量", self)
        self.distance_measure_action.setCheckable(True)
        self.distance_measure_action.setToolTip("点击启用后在图像上拖动可测量距离")
        self.distance_measure_action.triggered.connect(self.toggle_distance_measure)
        
        # 添加到工具菜单
        self.tool_menu.addAction(self.distance_measure_action)
        
        # 将菜单关联到工具按钮
        self.tool_action.setMenu(self.tool_menu)
        
        self.help_action = QAction("帮助", self)
        self.help_action.setToolTip("显示帮助")

        # 连接动作信号
        self.open_action.triggered.connect(self.handle_open)
        self.clear_action.triggered.connect(self.handle_clear)
        self.help_action.triggered.connect(self.show_help)

        # 添加动作到工具栏
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.clear_action)
        
        # 直接创建一个工具按钮，而不是使用addAction
        tool_button = QToolButton()
        tool_button.setText("工具")
        tool_button.setToolTip("工具选项")
        tool_button.setPopupMode(QToolButton.InstantPopup)  # 改回InstantPopup，任何点击都显示菜单
        tool_button.setMenu(self.tool_menu)
        # 使按钮样式与工具栏一致
        tool_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # 添加按钮到工具栏
        self.toolbar.addWidget(tool_button)
        
        self.toolbar.addAction(self.help_action)
        
        # 设置工具栏样式，确保按钮文字可见
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # 设置工具栏图标大小
        self.toolbar.setIconSize(QSize(24, 24))

    def toggle_distance_measure(self, checked):
        """切换距离测量工具的状态"""
        current_tab = self.tab_widget.currentWidget()
        # 检查当前标签页类型并启用或禁用测量工具
        if hasattr(current_tab, 'enable_distance_measure'):
            current_tab.enable_distance_measure(checked)
            if checked:
                self.status_bar.showMessage("距离测量工具已启用")
            else:
                self.status_bar.showMessage("距离测量工具已禁用")
                # 取消选中状态
                self.distance_measure_action.setChecked(False)

    def show_help(self):
        """显示帮助对话框"""
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

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
        # 更新工具栏按钮状态 - 让所有标签页都能使用Clear功能
        current_tab = self.tab_widget.widget(index)
        self.open_action.setEnabled(True)  # 所有标签页都可以打开文件
        self.clear_action.setEnabled(True)  # 所有标签页都可以清除内容
        
        # 切换标签页时取消选中测量工具
        if self.distance_measure_action.isChecked():
            self.distance_measure_action.setChecked(False)
            # 如果当前标签页支持测量功能，则禁用它
            if hasattr(current_tab, 'enable_distance_measure'):
                current_tab.enable_distance_measure(False)
        
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
            elif isinstance(current_tab, MedSAMTab):
                current_tab.load_image()
            elif isinstance(current_tab, VTKReconstructionTab):
                current_tab.load_dicom()
        except Exception as e:
            self.status_bar.showMessage(f"Error opening file: {str(e)}")

    def handle_clear(self):
        """处理清除图像/模型动作"""
        try:
            # 获取当前活动的标签页
            current_index = self.tab_widget.currentIndex()
            
            # 超分辨率标签页
            if current_index == 0:
                sr_tab = self.sr_tab
                # 检查是否有图像
                if hasattr(sr_tab, 'image') and sr_tab.image is not None:
                    # 清除原始图像标签
                    sr_tab.video_label.clear()
                    
                    # 使用image_viewer清除SR结果
                    if hasattr(sr_tab, 'image_viewer') and sr_tab.image_viewer is not None:
                        sr_tab.image_viewer.scene().clear()
                    
                    # 重置图像变量
                    sr_tab.image = None
                    sr_tab.sr_result = None
                    # 禁用保存按钮
                    sr_tab.save_button.setEnabled(False)
                    self.status_bar.showMessage("超分辨率图像已清除")
                else:
                    self.status_bar.showMessage("超分辨率标签页无图像可清除")
            
            # 分割标签页
            elif current_index == 1:
                seg_tab = self.seg_tab
                # 清除分割相关内容
                if hasattr(seg_tab, 'img_3c') and seg_tab.img_3c is not None:
                    # 清除场景
                    seg_tab.scene.clear()
                    # 重置图像和掩码变量
                    seg_tab.img_3c = None
                    seg_tab.mask_c = None
                    seg_tab.embedding = None
                    seg_tab.prev_mask = None
                    # 清除EDV和ESV标签
                    if hasattr(seg_tab, 'edv_label'):
                        seg_tab.edv_label.clear()
                    if hasattr(seg_tab, 'esv_label'):
                        seg_tab.esv_label.clear()
                    seg_tab.edv_mask = None
                    seg_tab.esv_mask = None
                    # 更新掩码状态
                    if hasattr(seg_tab, 'update_mask_status'):
                        seg_tab.update_mask_status()
                    self.status_bar.showMessage("分割数据已清除")
                else:
                    self.status_bar.showMessage("无分割数据可清除")
            
            # 3D重建标签页
            elif current_index == 2:
                recons_tab = self.recons_tab
                # 清除3D重建相关内容
                if hasattr(recons_tab, 'reader') and recons_tab.reader is not None:
                    # 重置读取器
                    recons_tab.reader = None
                    recons_tab.current_model = None
                    # 清除渲染器
                    if hasattr(recons_tab, 'renderer'):
                        recons_tab.renderer.RemoveAllViewProps()
                        # 重新添加FPS文本Actor
                        if hasattr(recons_tab, 'fps_text_actor') and recons_tab.fps_text_actor:
                            recons_tab.renderer.AddActor2D(recons_tab.fps_text_actor)
                        # 清除MPR三平面视图
                        if hasattr(recons_tab, 'mpr_axial_view') and recons_tab.mpr_axial_view:
                            recons_tab.mpr_axial_view.GetRenderer().RemoveAllViewProps()
                        if hasattr(recons_tab, 'mpr_coronal_view') and recons_tab.mpr_coronal_view:
                            recons_tab.mpr_coronal_view.GetRenderer().RemoveAllViewProps()
                        if hasattr(recons_tab, 'mpr_sagittal_view') and recons_tab.mpr_sagittal_view:
                            recons_tab.mpr_sagittal_view.GetRenderer().RemoveAllViewProps()
                        if hasattr(recons_tab.vtk_widget, 'GetRenderWindow'):
                            recons_tab.vtk_widget.GetRenderWindow().Render()
                    self.status_bar.showMessage("3D模型已清除")
                else:
                    self.status_bar.showMessage("无3D模型可清除")
        except Exception as e:
            self.status_bar.showMessage(f"清除内容时出错: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_()) 