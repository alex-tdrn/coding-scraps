find_package(Vulkan REQUIRED)
find_package(glfw REQUIRED)
find_package(glm REQUIRED)

find_program(GLSLC glslc REQUIRED)

#TODO make a custom target for building all shaders or something
function(add_shaders SH_FOLDER)
	file(GLOB SH_FILES
		RELATIVE ${SH_FOLDER}
		CONFIGURE_DEPENDS
		${SH_FOLDER}/*
	)
	foreach(SH_FILE ${SH_FILES})
	add_custom_command(
			TARGET 
			vulkan-hello-triangle
			POST_BUILD
			COMMAND 
			glslc ${SH_FOLDER}/${SH_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SH_FILE}.spv
		)
	endforeach()
endfunction()

add_subdirectory("hello-triangle")