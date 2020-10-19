add_executable(vulkan-generating-mipmaps "src/main.cpp" "src/GeneratingMipmaps.cpp" "inc/GeneratingMipmaps.h")

target_include_directories(vulkan-generating-mipmaps PUBLIC "inc")

target_add_shaders(vulkan-generating-mipmaps ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-generating-mipmaps Vulkan::Vulkan)
target_link_libraries(vulkan-generating-mipmaps glm::glm)
target_link_libraries(vulkan-generating-mipmaps glfw::glfw)
target_link_libraries(vulkan-generating-mipmaps stb::stb)
target_link_libraries(vulkan-generating-mipmaps tinyobjloader::tinyobjloader)
