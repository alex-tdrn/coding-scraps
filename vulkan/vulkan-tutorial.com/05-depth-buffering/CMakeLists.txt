add_executable(vulkan-depth-buffering "src/main.cpp" "src/DepthBuffering.cpp" "inc/DepthBuffering.h")

target_include_directories(vulkan-depth-buffering PUBLIC "inc")

target_add_shaders(vulkan-depth-buffering ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-depth-buffering Vulkan::Vulkan)
target_link_libraries(vulkan-depth-buffering glm::glm)
target_link_libraries(vulkan-depth-buffering glfw::glfw)
target_link_libraries(vulkan-depth-buffering stb::stb)
