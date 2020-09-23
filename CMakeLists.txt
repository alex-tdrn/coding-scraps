cmake_minimum_required(VERSION 3.18)

project(coding-scraps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_program(CONAN_FOUND conan REQUIRED)

execute_process(
	COMMAND conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build missing 
		-s build_type=${CMAKE_BUILD_TYPE} -e CXX=${CMAKE_CXX_COMPILER} -e CC=${CMAKE_C_COMPILER})
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

add_subdirectory("vulkan")