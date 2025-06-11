# deploy_core/CMakeLists.txt

# 这个库是 header-only, 所以它是一个 INTERFACE 库
add_library(deploy_core INTERFACE)

# 它没有自己的源文件或依赖，但它提供了头文件
# INTERFACE 关键字意味着任何链接到它的目标都会自动获得这个 include 目录
target_include_directories(deploy_core
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)