cmake_minimum_required(VERSION 3.18)

project(project-euler-solutions)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_program(CONAN_FOUND conan REQUIRED)

execute_process(
	COMMAND conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build missing -s build_type=${CMAKE_BUILD_TYPE}
)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

add_subdirectory("utility")
add_subdirectory("tests")
# add_subdirectory("solutions")
