cmake_minimum_required(VERSION 3.15)
project(detection_6d_foundationpose LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(assimp REQUIRED)
find_package(nvcv_types REQUIRED)
find_package(cvcuda REQUIRED)

file(GLOB_RECURSE SOURCES_CPP CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE SOURCES_CU CONFIGURE_DEPENDS src/*.cu)

add_library(detection_6d_foundationpose SHARED ${SOURCES_CPP} ${SOURCES_CU})

set_target_properties(detection_6d_foundationpose PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_property(TARGET detection_6d_foundationpose PROPERTY CUDA_ARCHITECTURES "86")

target_include_directories(detection_6d_foundationpose PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(detection_6d_foundationpose PUBLIC
        deploy_core
        foundationpose
        external_deps
        Eigen3::Eigen
        Threads::Threads
        assimp
        nvcv_types
        cvcuda
)