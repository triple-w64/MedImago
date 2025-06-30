# MedImago C++ 版本

MedImago是一个医学影像处理平台，提供超分辨率重建和三维重建功能。此版本使用C++重构，基于Qt、OpenCV和VTK实现。

## 功能特性

### 超分辨率重建模块

- 支持多种超分辨率算法 (EDSR, ESPCN, FSRCNN, EchoSR, LAPSRN)
- 图像浏览和管理
- 视频加载和处理
- 图像测量工具 (距离、角度)
- 图像处理功能 (窗位窗宽、亮度对比度调整等)

### 三维重建模块

- 支持DICOM数据加载和可视化
- 体绘制和表面绘制
- MPR多平面重建
- STL模型导出
- 光照和相机控制

## 依赖项

- Qt 5 (Core, Widgets, GUI, OpenGL)
- OpenCV 4+
- VTK 9+
- CMake 3.15+
- C++17 兼容编译器

## 构建步骤

### Linux下构建

```bash
# 安装依赖
sudo apt update
sudo apt install cmake g++ qtbase5-dev qtdeclarative5-dev libqt5opengl5-dev
sudo apt install libopencv-dev
sudo apt install libvtk9-dev

# 构建项目
mkdir build && cd build
cmake ..
make -j4

# 运行
./bin/MedImago
```

### Windows下构建

1. 安装[Visual Studio](https://visualstudio.microsoft.com/) 
2. 安装[CMake](https://cmake.org/download/)
3. 安装[Qt](https://www.qt.io/download)
4. 安装[OpenCV](https://opencv.org/releases/)
5. 安装[VTK](https://vtk.org/download/)

构建：
```
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="C:\Qt\5.15.2\msvc2019_64"
cmake --build . --config Release
```

## 使用说明

### 超分辨率重建

1. 打开图像或视频文件
2. 选择超分辨率算法和放大倍数
3. 点击"应用超分辨率"按钮
4. 保存处理结果

### 三维重建

1. 打开DICOM数据文件夹
2. 选择重建类型(体绘制、表面绘制或MPR)
3. 调整参数(窗位窗宽、不透明度等)
4. 点击"开始重建"按钮
5. 如需要，保存STL模型

## 许可证

GPL-3.0 license 