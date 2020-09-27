cmake_minimum_required(VERSION 3.18)

project(coding-scraps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_program(CONAN_FOUND conan REQUIRED)

execute_process(
	COMMAND conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build missing -s build_type=${CMAKE_BUILD_TYPE}
			-e CXX=${CMAKE_CXX_COMPILER} -e CC=${CMAKE_C_COMPILER}
)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

add_compile_definitions("$<$<CONFIG:DEBUG>:DEBUG>")

find_package(Vulkan REQUIRED)
find_package(glfw REQUIRED)
find_package(glm REQUIRED)

find_program(GLSLC glslc REQUIRED)

# TODO make a custom target for building all shaders or something
function(add_shaders SH_FOLDER)
	file(
		GLOB SH_FILES
		RELATIVE ${SH_FOLDER}
		CONFIGURE_DEPENDS ${SH_FOLDER}/*
	)
	foreach(SH_FILE ${SH_FILES})
		add_custom_command(
			TARGET vulkan-hello-triangle
			POST_BUILD
			COMMAND glslc ${SH_FOLDER}/${SH_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SH_FILE}.spv
		)
	endforeach()
endfunction()

add_subdirectory("hello-triangle")
