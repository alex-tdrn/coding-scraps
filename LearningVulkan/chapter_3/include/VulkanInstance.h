#pragma once
#include "VulkanLayerAndExtension.h"

#include <vector>
#include <vulkan/vulkan.h>

class VulkanInstance
{
public:
	VkInstance instance;
	VulkanLayerAndExtension layerExtension;

	VkResult createInstance(
		std::vector<const char*>& layers, std::vector<const char*>& extensions, const char* applicationName);
	void destroyInstance();
};