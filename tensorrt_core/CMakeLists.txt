# 1. 查找依赖项：这是关键的第一步
# 查找 CUDA 工具包，这是使用任何 CUDA API 的前提
find_package(CUDAToolkit REQUIRED)

# --- 库定义 ---
set(TENSORRT_CORE_SOURCES src/tensorrt_infer_core.cpp)
add_library(tensorrt_core SHARED ${TENSORRT_CORE_SOURCES})

# --- 头文件目录 ---
# 这部分保持不变，用于暴露此模块自己的头文件
target_include_directories(tensorrt_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# --- 链接所有依赖库 ---
# 2. 将 CUDA 和 TensorRT 库链接到你的目标
target_link_libraries(tensorrt_core PUBLIC
        # CUDA 运行时库。链接它会自动处理 CUDA 头文件路径
        CUDA::cudart

        # TensorRT核心库
        nvinfer

        # 如果你使用 ONNX 模型，则需要此库
        nvonnxparser

        # 你项目中其他的依赖
        deploy_core
        external_deps
)