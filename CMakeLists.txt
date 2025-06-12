cmake_minimum_required(VERSION 3.15)
project(deploy_foundationpose LANGUAGES CXX CUDA)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Find External Dependencies ---
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs NO_MODULE)
find_package(glog REQUIRED)

# --- "External Dependencies Bundle" ---
add_library(external_deps INTERFACE)
target_include_directories(external_deps INTERFACE
        $<BUILD_INTERFACE:${CMAKE_PROJECT_SOURCE_DIR}>
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(external_deps INTERFACE
        ${OpenCV_LIBRARIES}
        glog::glog
        nvinfer
        nvonnxparser
)

# --- Add Subprojects ---
add_subdirectory(deploy_core)
add_subdirectory(tensorrt_core)
add_subdirectory(foundationpose) # <-- Changed from foundationpose_lib

# --- Create Executable ---
add_executable(foundationpose_app main.cpp)

# --- Link Final Application ---
target_link_libraries(foundationpose_app PRIVATE
        foundationpose # <-- Changed from foundationpose_lib
)