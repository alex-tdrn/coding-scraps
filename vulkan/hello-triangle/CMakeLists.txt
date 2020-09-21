add_executable(
	vulkan-hello-triangle
	"src/main.cpp"
)

target_link_libraries(vulkan-hello-triangle Vulkan::Vulkan)
target_link_libraries(vulkan-hello-triangle glm::glm)
target_link_libraries(vulkan-hello-triangle glfw::glfw)

#TODO make a custom target for building all shaders or something
foreach(SHADERFILE shader.frag;shader.vert)
message("Compiling shader ${SHADERFILE}")
add_custom_command(
	TARGET 
		vulkan-hello-triangle
	POST_BUILD
 	COMMAND 
 		glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADERFILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SHADERFILE}.spv
)
endforeach()