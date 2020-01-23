#pragma once
#include <vector>
#include <vulkan/vulkan.h>

struct VulkanLayerProperties
{
	VkLayerProperties properties;
	std::vector<VkExtensionProperties> extensions;
};

class VulkanLayerAndExtension
{
private:
	std::vector<VulkanLayerProperties> layerPropertyList;

public:
	std::vector<const char*> appRequestedLayerNames;
	std::vector<const char*> appRequestedExtensionNames;

public:
	VkResult getInstanceLayerProperties();
	VkResult getExtensionProperties(VulkanLayerProperties& layerProps, VkPhysicalDevice* gpu = nullptr);
	VkResult getDeviceExtensionProperties(VkPhysicalDevice* gpu);
};