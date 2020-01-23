#include "VulkanApplication.h"

VulkanApplication* VulkanApplication::getInstance()
{
	static VulkanApplication instance;
	return &instance;
}

VulkanApplication::VulkanApplication()
{
	instanceObj.layerExtension.getInstanceLayerProperties();
}

VkResult VulkanApplication::createVulkanInstance(
	std::vector<const char*>& layers, std::vector<const char*>& extensions, const char* applicationName)
{
	instanceObj.createInstance(layers, extensions, applicationName);
	return VK_SUCCESS;
}