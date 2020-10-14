add_executable(vulkan-loading-models "src/main.cpp" "src/LoadingModels.cpp" "inc/LoadingModels.h")

target_include_directories(vulkan-loading-models PUBLIC "inc")

target_add_shaders(vulkan-loading-models ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-loading-models Vulkan::Vulkan)
target_link_libraries(vulkan-loading-models glm::glm)
target_link_libraries(vulkan-loading-models glfw::glfw)
target_link_libraries(vulkan-loading-models stb::stb)
