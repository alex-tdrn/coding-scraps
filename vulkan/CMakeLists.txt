find_package(Vulkan REQUIRED)
find_package(glfw REQUIRED)
find_package(glm REQUIRED)

find_program(GLSLC glslc REQUIRED)

add_subdirectory("hello-triangle")