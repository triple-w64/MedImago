import sys,re, os
from PyQt5.QtWidgets import QApplication, QWidget
from Ui_tagsys import Ui_MainWindow
from PyQt5.QtWidgets import QPushButton, QMainWindow, QLabel, QAction
from PyQt5.QtGui import QStandardItemModel, QStandardItem

# FileSystemModel
from PyQt5.QtWidgets import QFileSystemModel, QTreeView, QVBoxLayout

from PyQt5.QtCore import Qt, QModelIndex
import re
import win32clipboard as w

import ospath as op


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.setupMenubar()
        self.setMyCpu()
        self.setupFileFieldUi()
        self.setupStatusBarUi()

    def setupMenubar(self):
        action_debug = QAction('debug', self)
        action_debug.triggered.connect(self.debugWline)
        self.menu_4.addAction(action_debug)
    
    def setMyCpu(self):
        self.tabWidget.clear()
        self.tree_view = QTreeView(self)
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(1)
        self.tree_view.clicked.connect(self.tv_on_click)
        self.tree_view.expanded.connect(self.on_node_expanded)
        self.file_model = QFileSystemModel()
        self.fm = self.file_model
        # # 连接 directoryLoaded 信号到槽函数 on_directory_loaded
        # self.fm.directoryLoaded.connect(self.on_directory_loaded)

        # 创建一个带图标的根项目
        root_icon = self.fm.myComputer(1)
        self.root_item = QStandardItem(root_icon, '此电脑')  # 根项目没有名称
        self.root_item.setEditable(0)
        self.root_item.isloaded = True
        self.model.appendRow(self.root_item)
        # 添加自定义节点
        rootfolderlist = ["3D Objects", "Videos", "Pictures", "Documents", "Downloads", "Music", "Desktop"]
        rootfoldernames = ["3D对象", "视频", "图片", "文档", "下载", "音乐", "桌面"]
        for i in range(len(rootfolderlist)):
            user_folder = os.path.expanduser('~')
            itempath = os.path.join(user_folder, rootfolderlist[i])
            fmindex = self.fm.index(itempath)
            obj = self.fm.myComputer(5)
            icon = self.fm.fileIcon(fmindex)
            childitem = self.add_custom_node(self.root_item, rootfoldernames[i], itempath, icon)
            if not op.isempty(itempath): self.add_custom_node(childitem, rootfoldernames[i], itempath, icon)
            # self.add_custom_node(self.model.invisibleRootItem(), rootfoldernames[i], icon)

        # 添加盘符
        for drive in self.get_drives():
            # print(drive)
            driveindex = self.fm.index(drive)
            childitem = self.add_custom_node(self.root_item, self.fm.fileName(driveindex), drive, self.fm.fileIcon(self.fm.index(drive)))
            if not op.isempty(drive): self.add_custom_node(childitem, rootfoldernames[i], itempath, icon)
        self.tree_view.expand(self.model.indexFromItem(self.root_item))
        self.tabWidget.addTab(self.tree_view, "My computer")
    
    def setupFileFieldUi(self):
        ''''''
        self.tree_view2 = self.create_tree_view()
        # 将设置好的容器 widget 添加为一个新的 tab
        self.tabWidget_2.clear()
        self.tabWidget_2.addTab(self.tree_view2, "经典列表")
        self.pushButton.clicked.connect(self.retHigherLvDir)
        self.tree_view2.doubleClicked.connect(self.tv2_on_double_click)
        self.setRoot(self.fm.index(self.fm.myComputer()))
        
    def setRoot(self, index):
        if isinstance(index, QModelIndex):
            self.tree_view2.setRootIndex(index)
            # 检查新的父索引
            parent_index = self.fm.parent(index)
            mycpuindex = self.fm.index(self.fm.myComputer())
            if parent_index == index:
                print("yes")
            print("mycpuindex.isValid()", mycpuindex.isValid())
            # print("Name", self.fm.fileName(parent_index))
            # up_button
            self.pushButton.setEnabled(parent_index != index)
            self.tree_view2.collapseAll()
            curdir = self.status_bar.findChild(QLabel, "curdir")
            if curdir:
                self.status_bar.findChild(QLabel, "curdir").setText(self.fm.filePath(index))
    
    def retHigherLvDir(self):
        # 获取当前的索引
        current_index = self.tree_view2.rootIndex()
        # 获取父级索引
        parent_index = self.fm.parent(current_index)
        # 设置新的根路径
        self.setRoot(parent_index)
        
    def on_node_expanded(self, index):
        '''
        当节点被展开时，遍历其子节点，给其子节点加上子节点，并标记isloaded
        '''
        item = self.model.itemFromIndex(index)
        if item.isloaded:
            return
        if not item.hasChildren():
            return
        item.removeRow(0)

        childrenAbsPath = op.listdir(item.path)
        for childAbsPath in childrenAbsPath:
            childindex = self.fm.index(childAbsPath)
            childitem = self.add_custom_node(item, op.basename(childAbsPath),childAbsPath,self.fm.fileIcon(childindex))
            print("childAbsPath", childAbsPath)
            if not op.isempty(childAbsPath): self.add_custom_node(childitem, self.fm.fileName(childindex),childAbsPath,self.fm.fileIcon(childindex))
            # print(self.fm.fileName(childindex))
        # if item.data(Qt.UserRole) == 'Hello World':
        #     print('Hello World')
        item.isloaded = True
    
    def tv_on_click(self, index):
        # 获取被单击的项目
        item = self.model.itemFromIndex(index)
        self.setRoot(self.fm.index(item.path))
        
    def tv2_on_double_click(self, index):
        self.setRoot(index)
    
    def get_drives(self):
        return [f"{drive}:\\\\" for drive in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f"{drive}:")]
        
    def add_custom_node(self, parent, name, itempath, icon=None):
        item = QStandardItem(name)
        item.path = itempath
        item.isloaded = False
        item.setEditable(False)
        if icon:
            item.setIcon(icon)
            # print(type(icon))
        parent.appendRow(item)
        return item
    
    def debugWline(self):
        cmd = self.lineEdit.text()
        try:
            exec(cmd)
        except Exception as e:
            print(e)
    # def on_directory_loaded(self, path):
    #     index = self.fm.index(path)
    #     print(f"Directory loaded: {path}")
    #     # self.fm.setRootPath(path)
    #     self.setRoot(index)
    #     # 在这里执行加载目录后的操作
    #     # 例如，你可以展开节点、更新状态栏或执行其他任务
        
    def create_tree_view(self):
        # 创建一个新的 QTreeView 实例并设置模型等
        tree_view = QTreeView()
        self.fm.setRootPath('')
        tree_view.setModel(self.fm)
        return tree_view
    
    def setupStatusBarUi(self):
        # 创建并设置 QLabel 显示当前文件夹
        folder_label = QLabel("")
        folder_label.setObjectName("curdir")
        folder_label.setStyleSheet("margin-right: 10px;")  # 右侧间隔10px
        self.status_bar.addWidget(folder_label)

        # 创建并设置 QLabel 显示当前版本信息
        version_label = QLabel("版本 0.0.2")
        version_label.setStyleSheet("margin-left: 10px;")  # 右侧间隔10px
        self.status_bar.addPermanentWidget(version_label)  # 添加到状态栏的最右侧

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())