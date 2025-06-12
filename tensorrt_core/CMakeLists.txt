set(TENSORRT_CORE_SOURCES src/tensorrt_infer_core.cpp
src/tensorrt_blob_buffer.cpp src/tensorrt_infer_core_factory.cpp)
add_library(tensorrt_core SHARED ${TENSORRT_CORE_SOURCES})

target_include_directories(tensorrt_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tensorrt_core PUBLIC
        CUDA::cudart
        deploy_core
        external_deps
)