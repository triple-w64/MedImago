import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, 
    QTabWidget, QFileDialog, QSplitter, QWidget, QDesktopWidget,
    QTextBrowser, QVBoxLayout, QDialog
)
from PyQt5.QtCore import Qt, QCoreApplication
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
        readme_path = os.path.join("MedSAM", "README.md")
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
        
        # 设置窗口大小为1920x1080
        self.resize(1920, 1080)
        
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
        self.open_action = QAction("Open", self)
        self.clear_action = QAction("Clear", self)
        self.measure_action = QAction("Measure", self)
        self.tool_action = QAction("Tool", self)
        self.help_action = QAction("Help", self)

        self.open_action.triggered.connect(self.handle_open)
        self.clear_action.triggered.connect(self.handle_clear)
        self.help_action.triggered.connect(self.show_help)

        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.clear_action)
        self.toolbar.addAction(self.measure_action)
        self.toolbar.addAction(self.tool_action)
        self.toolbar.addAction(self.help_action)

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
        current_tab = self.tab_widget.widget(self.tab_widget.currentIndex())
        
        # 超分辨率标签页
        if isinstance(current_tab, SuperResolutionTab):
            # 检查是否有图像
            if current_tab.image is not None:
                # 清除原始图像标签
                current_tab.video_label.clear()
                
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
        
        # 分割标签页
        elif isinstance(current_tab, MedSAMTab):
            # 清除分割相关内容
            if current_tab.img_3c is not None:
                # 清除场景
                current_tab.scene.clear()
                # 重置图像和掩码变量
                current_tab.img_3c = None
                current_tab.mask_c = None
                current_tab.embedding = None
                current_tab.prev_mask = None
                # 清除EDV和ESV标签
                current_tab.edv_label.clear()
                current_tab.esv_label.clear()
                current_tab.edv_mask = None
                current_tab.esv_mask = None
                # 更新掩码状态
                current_tab.update_mask_status()
                self.status_bar.showMessage("Segmentation data cleared")
            else:
                self.status_bar.showMessage("No segmentation data to clear")
        
        # 3D重建标签页
        elif isinstance(current_tab, VTKReconstructionTab):
            # 清除3D重建相关内容
            if current_tab.reader is not None:
                # 重置读取器
                current_tab.reader = None
                current_tab.current_model = None
                # 清除渲染器
                if hasattr(current_tab, 'renderer'):
                    current_tab.renderer.RemoveAllViewProps()
                    if hasattr(current_tab.vtk_widget, 'GetRenderWindow'):
                        current_tab.vtk_widget.GetRenderWindow().Render()
                self.status_bar.showMessage("3D model cleared")
            else:
                self.status_bar.showMessage("No 3D model to clear")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_()) 