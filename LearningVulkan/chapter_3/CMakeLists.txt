cmake_minimum_required(VERSION 3.15)

project(chapter_3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Vulkan REQUIRED)

add_executable(chapter_3 
	"source/main.cpp"
	"source/VulkanApplication.cpp"
	"source/VulkanDevice.cpp"
	"source/VulkanInstance.cpp"
	"source/VulkanLayerAndExtension.cpp"
)

target_include_directories(chapter_3 
	PRIVATE 
	"include"
)

target_link_libraries(chapter_3 PRIVATE Vulkan::Vulkan)