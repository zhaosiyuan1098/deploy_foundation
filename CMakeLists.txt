cmake_minimum_required(VERSION 3.31)
project(deploy_foundationpose)

set(CMAKE_CXX_STANDARD 20)

add_executable(deploy_foundationpose main.cpp
        blockqueue.h)
