add_executable(
	vulkan-hello-triangle
	"src/main.cpp"
)

target_link_libraries(vulkan-hello-triangle Vulkan::Vulkan)
target_link_libraries(vulkan-hello-triangle glm::glm)
target_link_libraries(vulkan-hello-triangle glfw::glfw)

# add_custom_target(
# 	vulkan-hello-triangle-shaders
# 	COMMAND 
# 		glslc 

# )