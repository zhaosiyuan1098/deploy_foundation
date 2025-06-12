set(FOUNDATIONPOSE_SOURCES
        src/foundationpose.cpp
)

add_library(foundationpose SHARED ${FOUNDATIONPOSE_SOURCES})

target_include_directories(foundationpose PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(foundationpose PUBLIC
        tensorrt_core
)