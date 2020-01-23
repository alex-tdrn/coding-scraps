cmake_minimum_required(VERSION 3.15)

project(chapter_3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Vulkan REQUIRED)

add_executable(chapter_3 
	"source/main.cpp"

	"include/VulkanApplication.h"
	"source/VulkanApplication.cpp"

	"include/VulkanDevice.h"
	"source/VulkanDevice.cpp"

	"include/VulkanInstance.h"
	"source/VulkanInstance.cpp"

	"include/VulkanLayerAndExtension.h"
	"source/VulkanLayerAndExtension.cpp"
)

target_include_directories(chapter_3 
	PRIVATE 
	"include"
)

target_link_libraries(chapter_3 PRIVATE Vulkan::Vulkan)

target_compile_definitions(chapter_3 PRIVATE VK_USE_PLATFORM_WIN32_KHR)