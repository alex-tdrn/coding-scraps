add_executable(vulkan-multisampling "src/main.cpp" "src/Multisampling.cpp" "inc/Multisampling.h")

target_include_directories(vulkan-multisampling PUBLIC "inc")

target_add_shaders(vulkan-multisampling ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-multisampling Vulkan::Vulkan)
target_link_libraries(vulkan-multisampling glm::glm)
target_link_libraries(vulkan-multisampling glfw::glfw)
target_link_libraries(vulkan-multisampling stb::stb)
target_link_libraries(vulkan-multisampling tinyobjloader::tinyobjloader)
