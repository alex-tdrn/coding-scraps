#include "VulkanApplication.h"
#include "vulkan/vulkan_core.h"

std::vector<const char*> instanceExtensionNames = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};

std::vector<const char*> layerNames = {"VK_LAYER_LUNARG_api_dump"};

int main(int, char**)
{
	VulkanApplication* app = VulkanApplication::getInstance();
	app->createVulkanInstance(layerNames, instanceExtensionNames, "Hello world");
	return 0;
}