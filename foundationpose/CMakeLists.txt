# foundationpose/CMakeLists.txt

# 1. Define source files
set(FOUNDATIONPOSE_SOURCES
        src/foundationpose.cpp
)

# 2. Create the shared library with the new name
add_library(foundationpose SHARED ${FOUNDATIONPOSE_SOURCES}) # <-- Changed from foundationpose_lib

# 3. Specify include directories for the new target name
target_include_directories(foundationpose PUBLIC # <-- Changed from foundationpose_lib
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 4. Link dependencies to the new target name
target_link_libraries(foundationpose PUBLIC # <-- Changed from foundationpose_lib
        tensorrt_core
)