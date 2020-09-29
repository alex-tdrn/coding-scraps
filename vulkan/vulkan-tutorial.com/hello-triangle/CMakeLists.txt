add_executable(vulkan-hello-triangle "src/main.cpp" "src/HelloTriangle.cpp" "inc/HelloTriangle.h")

target_include_directories(vulkan-hello-triangle PUBLIC "inc")

target_add_shaders(vulkan-hello-triangle ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-hello-triangle Vulkan::Vulkan)
target_link_libraries(vulkan-hello-triangle glm::glm)
target_link_libraries(vulkan-hello-triangle glfw::glfw)
