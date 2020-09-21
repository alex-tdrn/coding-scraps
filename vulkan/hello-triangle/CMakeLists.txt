add_executable(
	vulkan-hello-triangle
	"src/main.cpp"
)

add_shaders(${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-hello-triangle Vulkan::Vulkan)
target_link_libraries(vulkan-hello-triangle glm::glm)
target_link_libraries(vulkan-hello-triangle glfw::glfw)
