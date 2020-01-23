#include "VulkanInstance.h"
#include "VulkanApplication.h"
#include "vulkan/vulkan_core.h"

VkResult VulkanInstance::createInstance(
	std::vector<const char*>& layers, std::vector<const char*>& extensions, const char* applicationName)
{
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pNext = nullptr;
	appInfo.pApplicationName = applicationName;
	appInfo.applicationVersion = 1;
	appInfo.pEngineName = applicationName;
	appInfo.engineVersion = 1;
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instInfo = {};
	instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instInfo.pNext = nullptr;
	instInfo.flags = 0;
	instInfo.pApplicationInfo = &appInfo;
	if(!layers.empty())
	{
		instInfo.enabledLayerCount = layers.size();
		instInfo.ppEnabledLayerNames = layers.data();
	}

	if(!extensions.empty())
	{
		instInfo.enabledExtensionCount = extensions.size();
		instInfo.ppEnabledExtensionNames = extensions.data();
	}

	return vkCreateInstance(&instInfo, nullptr, &instance);
}

void VulkanInstance::destroyInstance()
{
	vkDestroyInstance(instance, nullptr);
}
