#include "VulkanLayerAndExtension.h"

#include <iostream>
#include <string>

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

	std::cout << "Instance Layers(" << std::to_string(instanceLayerCount) << "):\n";

	for(auto layerProperty : layerProperties)
	{
		std::cout << "\n\t" << layerProperty.description << "  (" << layerProperty.layerName << ", "
				  << layerProperty.implementationVersion << ", " << layerProperty.specVersion << ")\n";

		VulkanLayerProperties prop;
		prop.properties = layerProperty;
		result = getExtensionProperties(prop);
		if(result)
			continue;

		layerPropertyList.push_back(prop);
		if(!prop.extensions.empty())
		{
			std::cout << "\tLayer Extensions(" << std::to_string(prop.extensions.size()) << "):\n";
			for(auto extension : prop.extensions)
				std::cout << "\t\t" << extension.extensionName << "  " << extension.specVersion << "\n";
		}
	}
	return result;
}

VkResult VulkanLayerAndExtension::getExtensionProperties(VulkanLayerProperties& layerProps, VkPhysicalDevice* gpu)
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

	return result;
}

VkResult VulkanLayerAndExtension::getDeviceExtensionProperties(VkPhysicalDevice* gpu)
{
	return VK_INCOMPLETE;
}