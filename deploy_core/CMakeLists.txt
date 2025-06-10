cmake_minimum_required(VERSION 3.31)
project(deploy_core)


add_compile_options(-std=c++20)
add_compile_options(-O1 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)

set(source_file main.cpp
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        glog::glog
)

target_include_directories(${PROJECT_NAME} PUBLIC include)


