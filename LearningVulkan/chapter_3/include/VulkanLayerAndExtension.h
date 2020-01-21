#pragma once
#include <vector>
#include <vulkan/vulkan.h>

struct LayerProperties
{
	VkLayerProperties properties;
	std::vector<VkExtensionProperties> extensions;
};

class VulkanLayerAndExtension
{
public:
	std::vector<LayerProperties> getInstanceLayerProperties();
	VkResult getExtensionProperties(LayerProperties& layerProps, VkPhysicalDevice* gpu = nullptr);
	VkResult getDeviceExtensionProperties(VkPhysicalDevice* gpu);
};