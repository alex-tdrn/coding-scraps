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
	VkResult getInstanceLayerProperties();
	VkResult getExtensionProperties(VulkanLayerProperties& layerProps, VkPhysicalDevice* gpu = nullptr);
	VkResult getDeviceExtensionProperties(VkPhysicalDevice* gpu);
};