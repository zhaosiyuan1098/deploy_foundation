# 主 CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(deploy_foundationpose LANGUAGES CXX CUDA)

# --- 全局设置 ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- 查找所有外部依赖项 ---
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs NO_MODULE)
find_package(glog REQUIRED)

# --- 创建“外部依赖项包裹” ---
add_library(external_deps INTERFACE)
target_include_directories(external_deps INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(external_deps INTERFACE
        ${OpenCV_LIBRARIES}
        glog::glog
        nvinfer
        nvonnxparser
)

# --- 添加所有子项目 ---
add_subdirectory(deploy_core)
add_subdirectory(triton_core)
add_subdirectory(foundationpose)

# --- 创建最终的可执行程序 ---
add_executable(foundationpose_app main.cpp)

# --- 链接最终程序 ---
# 只需要链接到最上层的算法库 foundationpose
# CMake 会自动处理传递依赖关系 (foundationpose -> triton_core -> ...)
target_link_libraries(foundationpose_app PRIVATE
        foundationpose_lib
)