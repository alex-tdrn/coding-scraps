cmake_minimum_required(VERSION 3.16)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.244.tar.gz"
    SHA1 "2c0f491fd0b80f7b09e3d21adb97237161ef9835"
)

project(coding-scraps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

hunter_add_package(glfw)
find_package(glfw3)

hunter_add_package(glm)
find_package(glm)

add_subdirectory("vulkan")