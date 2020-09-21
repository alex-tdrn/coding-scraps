cmake_minimum_required(VERSION 3.18)

project(coding-scraps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(glfw3)
find_package(glm)

add_subdirectory("vulkan")