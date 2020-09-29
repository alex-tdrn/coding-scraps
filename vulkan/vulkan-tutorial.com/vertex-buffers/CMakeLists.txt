add_executable(vulkan-vertex-buffers "src/main.cpp" "src/VertexBuffers.cpp" "inc/VertexBuffers.h")

target_include_directories(vulkan-vertex-buffers PUBLIC "inc")

target_add_shaders(vulkan-vertex-buffers ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-vertex-buffers Vulkan::Vulkan)
target_link_libraries(vulkan-vertex-buffers glm::glm)
target_link_libraries(vulkan-vertex-buffers glfw::glfw)
