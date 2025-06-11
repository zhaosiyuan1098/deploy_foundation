# triton_core/CMakeLists.txt

# 1. 定义源文件列表
set(TRITON_CORE_SOURCES
        src/triton_core.cpp
)

# 2. 创建一个共享库 (SHARED)
add_library(triton_core SHARED ${TRITON_CORE_SOURCES}
        )

# 3. 指定头文件目录
#    PUBLIC: 链接到 triton_core 的目标，以及 triton_core 自身，都能访问这个目录
target_include_directories(triton_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 4. 链接依赖项
#    PRIVATE: 这些依赖只在 triton_core.cpp 内部使用，不会传递给链接 triton_core 的目标
target_link_libraries(triton_core PRIVATE
        deploy_core       # 链接到 header-only 基础库
        external_deps     # 链接到包含 Triton, glog 等的外部依赖包裹
)