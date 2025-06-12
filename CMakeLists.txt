cmake_minimum_required(VERSION 3.15)

set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared" CACHE STRING "Choose between static or shared CUDA runtime library")
set(CUDA_USE_STATIC_LIBS OFF CACHE BOOL "Legacy flag to prefer dynamic CUDA runtime")

add_compile_options(-fvisibility=default)

project(deploy_foundationpose LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(glog REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(external_deps INTERFACE)

target_include_directories(external_deps INTERFACE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(external_deps INTERFACE
        ${OpenCV_LIBRARIES}
        glog::glog
        nvinfer
        nvonnxparser
)

add_subdirectory(deploy_core)
add_subdirectory(tensorrt_core)
add_subdirectory(foundationpose)
add_subdirectory(detection_6d_foundationpose)
add_subdirectory(simple_tests)

add_executable(foundationpose_app main.cpp)

target_link_libraries(foundationpose_app PRIVATE
        # 使用链接器组
        -Wl,--start-group
        foundationpose
        detection_6d_foundationpose
        tensorrt_core
        deploy_core
        -Wl,--end-group

        external_deps
)