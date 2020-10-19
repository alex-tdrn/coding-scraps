add_executable(vulkan-texture-mapping "src/main.cpp" "src/TextureMapping.cpp" "inc/TextureMapping.h")

target_include_directories(vulkan-texture-mapping PUBLIC "inc")

target_add_shaders(vulkan-texture-mapping ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

target_link_libraries(vulkan-texture-mapping Vulkan::Vulkan)
target_link_libraries(vulkan-texture-mapping glm::glm)
target_link_libraries(vulkan-texture-mapping glfw::glfw)
target_link_libraries(vulkan-texture-mapping stb::stb)
