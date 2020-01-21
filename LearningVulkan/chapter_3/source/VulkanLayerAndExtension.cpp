#include "VulkanLayerAndExtension.h"

#include <iostream>

VkResult VulkanLayerAndExtension::getInstanceLayerProperties()
{
	unsigned int instanceLayerCount;
	std::vector<VkLayerProperties> layerProperties;
	VkResult result;

	do
	{
		result = vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);

		if(result)
			return result;

		if(instanceLayerCount == 0)
			return VK_INCOMPLETE;

		layerProperties.resize(instanceLayerCount);
		result = vkEnumerateInstanceLayerProperties(&instanceLayerCount, layerProperties.data());

	} while(result == VK_INCOMPLETE);

	std::cout << "Instance Layers:\n\t";

	for(auto layerProperty : layerProperties)
	{
		std::cout << layerProperty.layerName << "  " << layerProperty.specVersion << "  "
				  << layerProperty.implementationVersion << "\n\t" << layerProperty.description << "\n\t";

		VulkanLayerProperties prop;
		prop.properties = layerProperty;
		result = getExtensionProperties(prop);
		if(result)
			continue;

		layerPropertyList.push_back(prop);
		std::cout << "Layer Extensions:\n";
		for(auto extension : prop.extensions)
			std::cout << "\t\t" << extension.extensionName << "  " << extension.specVersion << "\n";
	}
	return result;
}

VkResult VulkanLayerAndExtension::getExtensionProperties(
	VulkanLayerProperties& layerProps, VkPhysicalDevice* gpu = nullptr)
{
	unsigned int extensionCount;
	VkResult result;
	char* layerName = layerProps.properties.layerName;

	do
	{
		if(gpu)
			result = vkEnumerateDeviceExtensionProperties(*gpu, layerName, &extensionCount, nullptr);
		else
			result = vkEnumerateInstanceExtensionProperties(layerName, &extensionCount, nullptr);

		if(result || extensionCount == 0)
			continue;

		std::vector<VkExtensionProperties> extensionProperties;

		if(gpu)
			result = vkEnumerateDeviceExtensionProperties(*gpu, layerName, &extensionCount, extensionProperties.data());
		else
			result = vkEnumerateInstanceExtensionProperties(layerName, &extensionCount, extensionProperties.data());

	} while(result == VK_INCOMPLETE);
}

VkResult VulkanLayerAndExtension::getDeviceExtensionProperties(VkPhysicalDevice* gpu)
{
}