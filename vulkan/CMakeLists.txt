cmake_minimum_required(VERSION 3.18)

# TODO make this work on windows!!

project(coding-scraps-vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_program(CONAN_FOUND conan REQUIRED)

execute_process(
	COMMAND conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build missing -s build_type=${CMAKE_BUILD_TYPE}
)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

add_compile_definitions("$<$<CONFIG:DEBUG>:DEBUG>")

add_compile_definitions("GLM_FORCE_DEPTH_ZERO_TO_ONE")

find_package(Vulkan REQUIRED)
find_package(glfw REQUIRED)
find_package(glm REQUIRED)
find_package(stb REQUIRED)
find_package(tinyobjloader REQUIRED)

find_program(GLSLC glslc REQUIRED)

# TODO make a custom target for building all shaders or something
function(target_add_shaders TARGET SH_FOLDER)
	file(
		GLOB SH_FILES
		RELATIVE ${SH_FOLDER}
		CONFIGURE_DEPENDS ${SH_FOLDER}/*
	)
	foreach(SH_FILE ${SH_FILES})
		add_custom_command(
			TARGET ${TARGET}
			POST_BUILD
			COMMAND glslc ${SH_FOLDER}/${SH_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SH_FILE}.spv
		)
	endforeach()
endfunction()

add_subdirectory("vulkan-tutorial.com/01-hello-triangle")
add_subdirectory("vulkan-tutorial.com/02-vertex-buffers")
add_subdirectory("vulkan-tutorial.com/03-uniform-buffers")
add_subdirectory("vulkan-tutorial.com/04-texture-mapping")
add_subdirectory("vulkan-tutorial.com/05-depth-buffering")
add_subdirectory("vulkan-tutorial.com/06-loading-models")
add_subdirectory("vulkan-tutorial.com/07-generating-mipmaps")
add_subdirectory("vulkan-tutorial.com/08-multisampling")
