# foundationpose_lib/CMakeLists.txt

# 1. 定义源文件列表
set(FOUNDATIONPOSE_SOURCES
        src/foundationpose.cpp
)

# 2. 创建一个共享库
add_library(foundationpose_lib SHARED ${FOUNDATIONPOSE_SOURCES})

# 3. 指定头文件目录
target_include_directories(foundationpose_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 4. 链接依赖项
#    它依赖 triton_core 来执行推理
target_link_libraries(foundationpose_lib PRIVATE
        triton_core
)