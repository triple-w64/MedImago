import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, 
    QTabWidget, QFileDialog, QSplitter, QWidget, QDesktopWidget,
    QTextBrowser, QVBoxLayout, QDialog, QMenu, QToolButton,
    QDockWidget, QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout,
    QGroupBox
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

        # 创建共享文件浏览器
        self.create_file_browser()

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

    def create_file_browser(self):
        """创建共享文件浏览器"""
        # 创建文件浏览器
        file_dock = QDockWidget("文件浏览器", self)
        file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        file_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        
        file_widget = QWidget()
        file_layout = QVBoxLayout(file_widget)
        
        # 创建树形文件浏览器
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("文件")
        self.file_tree.itemDoubleClicked.connect(self.open_selected_file)
        
        # 添加按钮
        button_layout = QHBoxLayout()
        
        load_folder_btn = QPushButton("加载文件夹")
        load_folder_btn.clicked.connect(self.load_folder)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_file_tree)
        
        button_layout.addWidget(load_folder_btn)
        button_layout.addWidget(refresh_btn)
        
        # 添加到布局
        file_layout.addWidget(self.file_tree)
        file_layout.addLayout(button_layout)
        
        file_dock.setWidget(file_widget)
        
        # 添加到主窗口
        self.addDockWidget(Qt.LeftDockWidgetArea, file_dock)
        self.view_menu.addAction(file_dock.toggleViewAction())
    
    def load_folder(self):
        """加载文件夹并创建树形结构"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder_path:
            return
            
        self.file_tree.clear()
        root = QTreeWidgetItem(self.file_tree, [os.path.basename(folder_path)])
        root.setData(0, Qt.UserRole, folder_path)
        
        # 添加文件夹内容
        self._add_directory_contents(root, folder_path)
        root.setExpanded(True)
        
        # 通知标签页文件夹已加载
        if hasattr(self, 'sr_tab'):
            self.sr_tab.set_file_tree(self.file_tree)
        if hasattr(self, 'seg_tab'):
            self.seg_tab.set_file_tree(self.file_tree)
    
    def refresh_file_tree(self):
        """刷新文件树"""
        # 保存当前选择的文件夹路径
        current_root = None
        if self.file_tree.topLevelItemCount() > 0:
            root_item = self.file_tree.topLevelItem(0)
            current_root = root_item.data(0, Qt.UserRole)
            
        if current_root:
            self.file_tree.clear()
            root = QTreeWidgetItem(self.file_tree, [os.path.basename(current_root)])
            root.setData(0, Qt.UserRole, current_root)
            
            # 添加文件夹内容
            self._add_directory_contents(root, current_root)
            root.setExpanded(True)
    
    def _add_directory_contents(self, parent_item, parent_path):
        """添加文件夹内容到树形结构"""
        try:
            items = os.listdir(parent_path)
            
            # 先收集目录和文件
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
                # 只显示图像文件
                elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.dicom', '.dcm', '.mp4', '.avi', '.mkv')):
                    file_item = QTreeWidgetItem(None, [item])
                    file_item.setData(0, Qt.UserRole, full_path)
                    file_items.append(file_item)
            
            # 添加目录
            for dir_item, full_path in dir_items:
                parent_item.addChild(dir_item)
                dir_item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            
            # 添加文件
            for file_item in file_items:
                parent_item.addChild(file_item)
                
        except Exception as e:
            self.status_bar.showMessage(f"加载文件夹内容时出错: {str(e)}")
    
    def open_selected_file(self, item):
        """打开选中的文件"""
        file_path = item.data(0, Qt.UserRole)
        if os.path.isfile(file_path):
            # 根据当前标签页选择适当的加载方法
            current_tab = self.tab_widget.currentWidget()
            
            if isinstance(current_tab, SuperResolutionTab):
                # 检查文件类型
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    current_tab.load_image(file_path)
                elif file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
                    current_tab.load_video(file_path)
                else:
                    self.status_bar.showMessage(f"不支持的文件类型: {file_path}")
            
            elif hasattr(current_tab, 'load_specific_image'):
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    current_tab.load_specific_image(file_path)
                elif file_path.lower().endswith(('.dicom', '.dcm')):
                    self.status_bar.showMessage("正在加载DICOM文件...")
                    # 可添加DICOM加载逻辑
                else:
                    self.status_bar.showMessage(f"不支持的文件类型: {file_path}")
            else:
                self.status_bar.showMessage(f"当前标签页无法加载文件: {file_path}")
        else:
            # 如果是目录，展开它
            item.setExpanded(not item.isExpanded())
            
            # 如果是首次展开，加载内容
            if item.childCount() == 0:
                self._add_directory_contents(item, file_path)

    def create_actions(self):
        """创建工具栏动作"""
        # 创建动作并添加图标
        self.open_action = QAction(QIcon("resource/icons/open.png"), "打开", self)
        self.open_action.setToolTip("打开文件")
        
        self.clear_action = QAction(QIcon("resource/icons/clear.png"), "清除", self)
        self.clear_action.setToolTip("清除当前内容")
        
        # 添加指针工具
        self.pointer_action = QAction(QIcon("resource/icons/pointer.png"), "指针", self)
        self.pointer_action.setToolTip("切换到指针工具 (Ctrl+1)")
        self.pointer_action.setCheckable(True)
        self.pointer_action.setChecked(True)  # 默认选中
        self.pointer_action.triggered.connect(lambda checked: self.toggle_pointer_tool(checked))
        
        # 添加手型工具
        self.hand_action = QAction(QIcon("resource/icons/hand.png"), "手型", self)
        self.hand_action.setToolTip("切换到手型工具 (Ctrl+2)")
        self.hand_action.setCheckable(True)
        self.hand_action.triggered.connect(lambda checked: self.toggle_hand_tool(checked))
        
        # 工具菜单相关
        self.tool_action = QAction(QIcon("resource/icons/tools.png"), "工具", self)
        self.tool_action.setToolTip("工具选项")
        
        # 创建工具菜单
        self.tool_menu = QMenu(self)
        
        # 距离测量工具移到工具菜单中
        self.distance_measure_action = QAction(QIcon("resource/icons/measure_distance.png"), "距离测量", self)
        self.distance_measure_action.setCheckable(True)
        self.distance_measure_action.setToolTip("点击启用后在图像上拖动可测量距离 (Ctrl+3)")
        self.distance_measure_action.triggered.connect(self.toggle_distance_measure)
        
        # 添加角度测量工具
        self.angle_measure_action = QAction(QIcon("resource/icons/measure_angle.png"), "角度测量", self)
        self.angle_measure_action.setCheckable(True)
        self.angle_measure_action.setToolTip("点击启用后依次点击三个点以测量角度(中间点为角的顶点) (Ctrl+4)")
        self.angle_measure_action.triggered.connect(self.toggle_angle_measure)
        
        # 添加清除测量工具
        self.clear_measure_action = QAction(QIcon("resource/icons/clear_measure.png"), "清除测量", self)
        self.clear_measure_action.setToolTip("清除所有测量线和文本")
        self.clear_measure_action.triggered.connect(self.clear_measurements)
        
        # 添加撤销测量快捷键提示
        self.undo_measure_action = QAction(QIcon("resource/icons/undo.png"), "撤销测量 (Ctrl+Z)", self)
        self.undo_measure_action.setToolTip("撤销上一步测量")
        self.undo_measure_action.triggered.connect(self.undo_last_measurement)
        
        # 创建图像操作菜单
        self.image_menu = QMenu("图像操作")
        
        # 窗位窗宽调整
        self.window_level_action = QAction(QIcon("resource/icons/window_level.png"), "窗位窗宽调整", self)
        self.window_level_action.setToolTip("调整图像窗位窗宽以改变灰度显示范围")
        self.window_level_action.triggered.connect(self.adjust_window_level)
        
        # 亮度对比度调整
        self.brightness_contrast_action = QAction(QIcon("resource/icons/brightness.png"), "亮度/对比度调整", self)
        self.brightness_contrast_action.setToolTip("调整图像亮度和对比度")
        self.brightness_contrast_action.triggered.connect(self.adjust_brightness_contrast)
        
        # 图像增强选项
        self.enhancement_menu = QMenu("图像增强")
        self.enhancement_menu.setIcon(QIcon("resource/icons/enhance.png"))
        
        # 锐化
        self.sharpen_action = QAction(QIcon("resource/icons/sharpen.png"), "锐化增强", self)
        self.sharpen_action.triggered.connect(self.apply_sharpen)
        self.enhancement_menu.addAction(self.sharpen_action)
        
        # 平滑
        self.smooth_action = QAction(QIcon("resource/icons/smooth.png"), "平滑滤波", self)
        self.smooth_action.triggered.connect(self.apply_smooth)
        self.enhancement_menu.addAction(self.smooth_action)
        
        # 直方图均衡化
        self.histogram_eq_action = QAction(QIcon("resource/icons/histogram.png"), "直方图均衡化", self)
        self.histogram_eq_action.triggered.connect(self.apply_histogram_eq)
        self.enhancement_menu.addAction(self.histogram_eq_action)
        
        # 边缘检测
        self.edge_detection_action = QAction(QIcon("resource/icons/edge.png"), "边缘检测", self)
        self.edge_detection_action.triggered.connect(self.apply_edge_detection)
        self.enhancement_menu.addAction(self.edge_detection_action)
        
        # 图像变换选项
        self.transform_menu = QMenu("图像变换")
        self.transform_menu.setIcon(QIcon("resource/icons/transform.png"))
        
        # 水平翻转
        self.flip_h_action = QAction(QIcon("resource/icons/flip_h.png"), "水平翻转", self)
        self.flip_h_action.triggered.connect(self.apply_flip_horizontal)
        self.transform_menu.addAction(self.flip_h_action)
        
        # 垂直翻转
        self.flip_v_action = QAction(QIcon("resource/icons/flip_v.png"), "垂直翻转", self)
        self.flip_v_action.triggered.connect(self.apply_flip_vertical)
        self.transform_menu.addAction(self.flip_v_action)
        
        # 旋转90度
        self.rotate_90_action = QAction(QIcon("resource/icons/rotate.png"), "旋转90度", self)
        self.rotate_90_action.triggered.connect(self.apply_rotate_90)
        self.transform_menu.addAction(self.rotate_90_action)
        
        # ROI提取工具
        self.roi_extraction_action = QAction(QIcon("resource/icons/roi.png"), "ROI提取", self)
        self.roi_extraction_action.setToolTip("框选图像区域进行提取")
        self.roi_extraction_action.triggered.connect(self.extract_roi)
        
        # 添加所有操作到图像菜单
        self.image_menu.addAction(self.window_level_action)
        self.image_menu.addAction(self.brightness_contrast_action)
        self.image_menu.addMenu(self.enhancement_menu)
        self.image_menu.addMenu(self.transform_menu)
        self.image_menu.addAction(self.roi_extraction_action)
        
        # 添加到工具菜单
        self.tool_menu.addAction(self.distance_measure_action)
        self.tool_menu.addAction(self.angle_measure_action)
        self.tool_menu.addSeparator()
        self.tool_menu.addMenu(self.image_menu)
        self.tool_menu.addSeparator()
        self.tool_menu.addAction(self.clear_measure_action)
        self.tool_menu.addAction(self.undo_measure_action)
        
        # 将菜单关联到工具按钮
        self.tool_action.setMenu(self.tool_menu)
        
        self.help_action = QAction(QIcon("resource/icons/help.png"), "帮助", self)
        self.help_action.setToolTip("显示帮助")

        # 连接动作信号
        self.open_action.triggered.connect(self.handle_open)
        self.clear_action.triggered.connect(self.handle_clear)
        self.help_action.triggered.connect(self.show_help)

        # 添加动作到工具栏
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.clear_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.pointer_action)
        self.toolbar.addAction(self.hand_action)
        self.toolbar.addSeparator()
        
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
        # 设置工具栏文字颜色为深灰色
        self.toolbar.setStyleSheet("QToolButton { color: #444444; }")

    def toggle_pointer_tool(self, checked):
        """切换指针工具的状态"""
        if checked:
            current_tab = self.tab_widget.currentWidget()
            # 取消选中其他工具
            self.hand_action.setChecked(False)
            self.distance_measure_action.setChecked(False)
            self.angle_measure_action.setChecked(False)
            
            # 检查当前标签页类型并启用指针工具
            if hasattr(current_tab, 'switch_tool'):
                current_tab.switch_tool("pointer")
                self.status_bar.showMessage("指针工具已启用")

    def toggle_hand_tool(self, checked):
        """切换手型工具的状态"""
        if checked:
            current_tab = self.tab_widget.currentWidget()
            # 取消选中其他工具
            self.pointer_action.setChecked(False)
            self.distance_measure_action.setChecked(False)
            self.angle_measure_action.setChecked(False)
            
            # 检查当前标签页类型并启用手型工具
            if hasattr(current_tab, 'switch_tool'):
                current_tab.switch_tool("hand")
                self.status_bar.showMessage("手型工具已启用")

    def toggle_distance_measure(self, checked):
        """切换距离测量工具的状态"""
        current_tab = self.tab_widget.currentWidget()
        # 检查当前标签页类型并启用或禁用测量工具
        if hasattr(current_tab, 'enable_distance_measure'):
            current_tab.enable_distance_measure(checked)
            
            if checked:
                # 取消选中其他工具
                self.pointer_action.setChecked(False)
                self.hand_action.setChecked(False)
                self.angle_measure_action.setChecked(False)
                
                self.status_bar.showMessage("距离测量工具已启用")
            else:
                self.status_bar.showMessage("距离测量工具已禁用")
                # 取消选中状态
                self.distance_measure_action.setChecked(False)
                # 恢复指针工具
                self.pointer_action.setChecked(True)
                if hasattr(current_tab, 'switch_tool'):
                    current_tab.switch_tool("pointer")
    
    def toggle_angle_measure(self, checked):
        """切换角度测量工具的状态"""
        current_tab = self.tab_widget.currentWidget()
        # 检查当前标签页类型并启用或禁用角度测量工具
        if hasattr(current_tab, 'enable_angle_measure'):
            current_tab.enable_angle_measure(checked)
            
            if checked:
                # 取消选中其他工具
                self.pointer_action.setChecked(False)
                self.hand_action.setChecked(False)
                self.distance_measure_action.setChecked(False)
                
                self.status_bar.showMessage("角度测量工具已启用")
            else:
                self.status_bar.showMessage("角度测量工具已禁用")
                # 取消选中状态
                self.angle_measure_action.setChecked(False)
                # 恢复指针工具
                self.pointer_action.setChecked(True)
                if hasattr(current_tab, 'switch_tool'):
                    current_tab.switch_tool("pointer")
    
    def undo_last_measurement(self):
        """撤销上一步测量"""
        current_tab = self.tab_widget.currentWidget()
        # 检查当前标签页类型并调用相应的撤销方法
        if hasattr(current_tab, 'undo_last_measurement'):
            current_tab.undo_last_measurement()
    
    def clear_measurements(self):
        """清除所有测量"""
        current_tab = self.tab_widget.currentWidget()
        # 检查当前标签页类型并清除测量
        if hasattr(current_tab, 'clear_measurement'):
            current_tab.clear_measurement()
            
            # 取消选中测量工具
            if self.distance_measure_action.isChecked():
                self.distance_measure_action.setChecked(False)
            if self.angle_measure_action.isChecked():
                self.angle_measure_action.setChecked(False)
            
            # 恢复指针工具
            self.pointer_action.setChecked(True)
            if hasattr(current_tab, 'switch_tool'):
                current_tab.switch_tool("pointer")

    def show_help(self):
        """显示帮助对话框"""
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

    def create_tabs(self):
        """创建标签页"""
        # 标签页名称
        self.tab_names = ["Super-Resolution", "Segmentation", "Static 3D Reconstruction"]
        
        # 使用可停靠部件代替普通标签
        self.dockable_widgets = {}
        
        # 直接创建所有标签页（不使用懒加载，避免切换问题）
        self.sr_tab = SuperResolutionTab(self, self.status_bar)
        self.sr_tab.set_file_tree(self.file_tree)  # 设置共享文件树
        self.tab_widget.addTab(self.sr_tab, self.tab_names[0])
        
        self.seg_tab = MedSAMTab(self.status_bar, self)
        self.seg_tab.set_file_tree(self.file_tree)  # 设置共享文件树
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
        
        # 检查当前标签页类型，更新工具栏按钮状态
        if index == 0:  # 超分辨率标签页
            # 启用工具栏按钮
            if hasattr(self, 'pointer_action'):
                self.pointer_action.setEnabled(True)
                self.pointer_action.setChecked(True)
            if hasattr(self, 'hand_action'):
                self.hand_action.setEnabled(True)
                self.hand_action.setChecked(False)
            # 启用测量工具
            self.distance_measure_action.setEnabled(True)
            self.angle_measure_action.setEnabled(True)
        elif index == 1:  # 分割标签页
            # 禁用指针和手型工具，这些工具在分割页面会干扰绘制矩形框
            if hasattr(self, 'pointer_action'):
                self.pointer_action.setEnabled(False)
                self.pointer_action.setChecked(False)
            if hasattr(self, 'hand_action'):
                self.hand_action.setEnabled(False)
                self.hand_action.setChecked(False)
            # 禁用测量工具
            self.distance_measure_action.setChecked(False)
            self.distance_measure_action.setEnabled(False)
            self.angle_measure_action.setChecked(False)
            self.angle_measure_action.setEnabled(False)
        elif index == 2:  # 3D重建标签页
            # 在3D标签页中启用指针和手型工具
            if hasattr(self, 'pointer_action'):
                self.pointer_action.setEnabled(True)
                self.pointer_action.setChecked(True)
            if hasattr(self, 'hand_action'):
                self.hand_action.setEnabled(True)
                self.hand_action.setChecked(False)
            # 禁用测量工具（因为3D重建页面使用不同的交互方式）
            self.distance_measure_action.setChecked(False)
            self.distance_measure_action.setEnabled(False)
            self.angle_measure_action.setChecked(False)
            self.angle_measure_action.setEnabled(False)
        
        # 切换标签页时取消选中测量工具
        if self.distance_measure_action.isChecked():
            self.distance_measure_action.setChecked(False)
            # 如果当前标签页支持测量功能，则禁用它
            if hasattr(current_tab, 'enable_distance_measure'):
                current_tab.enable_distance_measure(False)
                
        # 切换标签页时取消选中角度测量工具
        if self.angle_measure_action.isChecked():
            self.angle_measure_action.setChecked(False)
            # 如果当前标签页支持角度测量功能，则禁用它 
            if hasattr(current_tab, 'enable_angle_measure'):
                current_tab.enable_angle_measure(False)
        
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

    def adjust_window_level(self):
        """调整窗位窗宽"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'adjust_window_level'):
            current_tab.adjust_window_level()
            self.status_bar.showMessage("窗位窗宽调整工具已启动")
    
    def adjust_brightness_contrast(self):
        """调整亮度对比度"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'adjust_brightness_contrast'):
            current_tab.adjust_brightness_contrast()
            self.status_bar.showMessage("亮度/对比度调整工具已启动")
    
    def apply_sharpen(self):
        """应用锐化滤波"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_sharpen'):
            current_tab.apply_sharpen()
            self.status_bar.showMessage("已应用锐化增强")
    
    def apply_smooth(self):
        """应用平滑滤波"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_smooth'):
            current_tab.apply_smooth()
            self.status_bar.showMessage("已应用平滑滤波")
    
    def apply_histogram_eq(self):
        """应用直方图均衡化"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_histogram_eq'):
            current_tab.apply_histogram_eq()
            self.status_bar.showMessage("已应用直方图均衡化")
    
    def apply_edge_detection(self):
        """应用边缘检测"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_edge_detection'):
            current_tab.apply_edge_detection()
            self.status_bar.showMessage("已应用边缘检测")
    
    def apply_flip_horizontal(self):
        """水平翻转图像"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_flip_horizontal'):
            current_tab.apply_flip_horizontal()
            self.status_bar.showMessage("已水平翻转图像")
    
    def apply_flip_vertical(self):
        """垂直翻转图像"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_flip_vertical'):
            current_tab.apply_flip_vertical()
            self.status_bar.showMessage("已垂直翻转图像")
    
    def apply_rotate_90(self):
        """旋转图像90度"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'apply_rotate_90'):
            current_tab.apply_rotate_90()
            self.status_bar.showMessage("已旋转图像90度")
    
    def extract_roi(self):
        """提取感兴趣区域"""
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, 'extract_roi'):
            current_tab.extract_roi()
            self.status_bar.showMessage("ROI提取工具已启动")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_()) 