cmake_minimum_required(VERSION 3.15)
project(simple_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)

# 最佳实践：明确列出源文件，而不是使用 GLOB，这样更稳定可靠
set(SIMPLE_TESTS_SOURCES
        src/test_foundationpose.cpp
)

add_executable(${PROJECT_NAME} ${SIMPLE_TESTS_SOURCES})

# 对于可执行文件，包含目录通常是 PRIVATE，因为它不需要向外传递
target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 对于可执行文件，链接也应该是 PRIVATE
# 链接顺序至关重要：高层库在前，底层库在后
target_link_libraries(${PROJECT_NAME} PRIVATE
        GTest::gtest_main

        # --- 使用链接器组来解决循环依赖 ---
        -Wl,--start-group
        foundationpose
        detection_6d_foundationpose
        tensorrt_core
        deploy_core
        -Wl,--end-group
        # --- 组结束 ---

        external_deps
)


# 启用并发现 GTest 测试
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})