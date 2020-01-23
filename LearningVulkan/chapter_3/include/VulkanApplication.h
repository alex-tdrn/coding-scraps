#pragma once

#include "VulkanInstance.h"
#include "vulkan/vulkan_core.h"

class VulkanApplication
{
private:
	VulkanApplication();

public:
	~VulkanApplication() = default;
	static VulkanApplication* getInstance();

	VkResult createVulkanInstance(
		std::vector<const char*>& layers, std::vector<const char*>& extensions, const char* applicationName);

	VulkanInstance instanceObj;
};