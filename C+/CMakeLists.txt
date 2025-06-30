cmake_minimum_required(VERSION 3.15)
project(MedImago VERSION 1.0.0 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 输出文件夹设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找依赖包
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

# 资源文件
set(RESOURCE_FILES
    resources/resources.qrc
)

# 源文件列表
set(SOURCE_FILES
    src/main.cpp
    src/MainWindow.cpp
    src/SuperResolutionTab.cpp
    src/ReconstructionTab.cpp
)

# 头文件列表
set(HEADER_FILES
    include/MainWindow.h
    include/SuperResolutionTab.h
    include/ReconstructionTab.h
)

# 添加可执行文件
add_executable(MedImago ${SOURCE_FILES} ${HEADER_FILES} ${RESOURCE_FILES})

# 包含目录
target_include_directories(MedImago PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接依赖库
target_link_libraries(MedImago
    Qt5::Widgets
    Qt5::Core
    Qt5::Gui
    -lGL
)

# Windows特定设置
if(WIN32)
    set_target_properties(MedImago PROPERTIES WIN32_EXECUTABLE TRUE)
endif()

# 安装规则
install(TARGETS MedImago DESTINATION bin)

# 创建目录结构
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources) 