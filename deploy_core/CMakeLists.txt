# deploy_core/CMakeLists.txt (修正后)

file(GLOB DEPLOY_CORE_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

add_library(deploy_core SHARED ${DEPLOY_CORE_SOURCES})

target_include_directories(deploy_core
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# --- 添加下面这部分 ---
# 将 deploy_core 链接到在主 CMakeLists.txt 中定义的 "external_deps"
# 这会把 OpenCV, glog 等依赖项的头文件路径和库正确地传递给 deploy_core
target_link_libraries(deploy_core
        PUBLIC
        external_deps
)