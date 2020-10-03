add_executable(vulkan-uniform-buffers "src/main.cpp" "src/UniformBuffers.cpp" "inc/UniformBuffers.h")

target_include_directories(vulkan-uniform-buffers PUBLIC "inc")

target_add_shaders(vulkan-uniform-buffers ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-uniform-buffers Vulkan::Vulkan)
target_link_libraries(vulkan-uniform-buffers glm::glm)
target_link_libraries(vulkan-uniform-buffers glfw::glfw)
