import sys,re, os
from PyQt5.QtWidgets import QApplication, QWidget
from Ui_tagsys import Ui_MainWindow
from PyQt5.QtWidgets import QPushButton, QMainWindow, QAction
from PyQt5.QtGui import QStandardItemModel, QStandardItem

# FileSystemModel
from PyQt5.QtWidgets import QFileSystemModel, QTreeView, QVBoxLayout

from PyQt5.QtCore import Qt
import re
import win32clipboard as w

import ospath as op


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.setMyCpu()
        
        self.tree_view2 = self.create_tree_view()
        # 将设置好的容器 widget 添加为一个新的 tab
        self.tabWidget_2.clear()
        self.tabWidget_2.addTab(self.tree_view2, "")
        
        action_hello = QAction('Print Hello World', self)
        action_hello.triggered.connect(self.print_hello)
        self.menu_4.addAction(action_hello)
    
    def setMyCpu(self):
        self.tabWidget.clear()
        self.tree_view = QTreeView(self)
        self.model = QStandardItemModel()
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(1)
        self.tree_view.doubleClicked.connect(self.tv_on_double_click)
        self.file_model = QFileSystemModel()
        self.fm = self.file_model
        # 连接 directoryLoaded 信号到槽函数 on_directory_loaded
        self.fm.directoryLoaded.connect(self.on_directory_loaded)

        # 创建一个带图标的根项目
        root_icon = self.fm.myComputer(1)
        self.root_item = QStandardItem(root_icon, '此电脑')  # 根项目没有名称
        self.root_item.setEditable(0)
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
            self.add_custom_node(self.root_item, rootfoldernames[i], itempath, icon)
            # self.add_custom_node(self.model.invisibleRootItem(), rootfoldernames[i], icon)

        # 添加盘符
        for drive in self.get_drives():
            # print(drive)
            driveindex = self.fm.index(drive)
            self.add_custom_node(self.root_item, self.fm.fileName(driveindex), drive, self.fm.fileIcon(self.fm.index(drive)))
        self.tree_view.expand(self.model.indexFromItem(self.root_item))
        self.tabWidget.addTab(self.tree_view, "My computer")
    
    def tv_on_double_click(self, index):
        # 获取被双击的项目
        item = self.model.itemFromIndex(index)
        # print(item.path)
        if op.isfile(item.path):
            return

        # set list tree view root index
        # self.on_directory_loaded(item.path)
        self.tree_view2.setRootIndex(self.fm.index(item.path))

        if item.ismounted:
            return
        # add children node
        childrenAbsPath = op.listdir(item.path)
        for childAbsPath in childrenAbsPath:
            childindex = self.fm.index(childAbsPath)
            self.add_custom_node(item, self.fm.fileName(childindex),childAbsPath,self.fm.fileIcon(childindex))
            # print(self.fm.fileName(childindex))
        # if item.data(Qt.UserRole) == 'Hello World':
        #     print('Hello World')
        item.ismounted = True
        
        
    
    def get_drives(self):
        return [f"{drive}:\\\\" for drive in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f"{drive}:")]
        
    def add_custom_node(self, parent, name, itempath, icon=None):
        item = QStandardItem(name)
        item.path = itempath
        item.ismounted = False
        item.setEditable(False)
        if icon:
            item.setIcon(icon)
            # print(type(icon))
        parent.appendRow(item)
    
    def print_hello(self):
        print(self.fm.index(""))
        pass
    def on_directory_loaded(self, path):
        index = self.fm.index(path)
        print(f"Directory loaded: {path}")
        # self.fm.setRootPath(path)
        self.tree_view2.setRootIndex(index)
        # 在这里执行加载目录后的操作
        # 例如，你可以展开节点、更新状态栏或执行其他任务
        
        
    def create_tree_view(self):
        # 创建一个新的 QTreeView 实例并设置模型等
        tree_view = QTreeView()
        self.fm.setRootPath('')
        tree_view.setModel(self.fm)
        # tree_view.setRootIndex(file_model.index(""))
        return tree_view
    

        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())