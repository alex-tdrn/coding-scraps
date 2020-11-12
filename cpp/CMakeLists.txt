cmake_minimum_required(VERSION 3.18)

project(coding-scraps-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	message(FATAL_ERROR "Clang required")
endif()

add_compile_options(-stdlib=libc++ -fimplicit-modules -fimplicit-module-maps)
add_link_options(-stdlib=libc++ -lc++abi)

add_subdirectory("modules")
