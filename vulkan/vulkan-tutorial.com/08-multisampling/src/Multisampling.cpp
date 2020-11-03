#include "Multisampling.h"

#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <set>
#include <stb_image.h>
#include <stdexcept>
#include <tiny_obj_loader.h>
#include <unordered_map>

#ifdef DEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif

const std::string meshPath = "../../../../meshes/viking_room.obj";
const std::string texturePath = "../../../../textures/viking_room.png";

PFN_vkCreateDebugUtilsMessengerEXT pfnVkCreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT pfnVkDestroyDebugUtilsMessengerEXT;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pMessenger)
{
	return pfnVkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
	VkInstance instance, VkDebugUtilsMessengerEXT messenger, VkAllocationCallbacks const* pAllocator)
{
	return pfnVkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	Multisampling* app = reinterpret_cast<Multisampling*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}

void keypressCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

std::vector<char> readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	std::size_t fileSize = file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);

	if(!file.is_open())
		throw std::runtime_error("failed to open file " + filename);

	return buffer;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::cerr << "Validation layer:\n"
			  << std::to_string(pCallbackData->messageIdNumber) << " " << pCallbackData->pMessageIdName << '\n'
			  << pCallbackData->pMessage << "\n\n";

	return false;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	Multisampling* app = reinterpret_cast<Multisampling*>(glfwGetWindowUserPointer(window));

	float f = 1.0f;

	if(yoffset < 0)
	{
		f += 0.1;
	}
	else if(yoffset > 0)
	{
		f -= 0.1;
	}

	app->targetFOV = std::min(std::max(app->targetFOV * f, 1.0f), 120.0f);
}

void Multisampling::run()
{
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}

void Multisampling::initWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

	glfwSetWindowUserPointer(window, this);

	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	glfwSetKeyCallback(window, keypressCallback);
	glfwSetScrollCallback(window, scroll_callback);
}

void Multisampling::initVulkan()
{
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createCommandPool();
	createDepthResources();
	createFramebuffers();
	loadModel();
	createVertexBuffer();
	createTextureImage();
	createTextureImageView();
	createTextureSampler();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();
	createSyncObjects();
}

void Multisampling::createInstance()
{
	if(enableValidationLayers && !checkValidationLayerSupport())
		throw std::runtime_error("validation layers requested, but not available!");

	auto appInfo = vk::ApplicationInfo().setPApplicationName("Hello Triangle").setApiVersion(VK_API_VERSION_1_2);

	auto extensions = getRequiredExtensions();
	auto instanceInfo = vk::InstanceCreateInfo().setPApplicationInfo(&appInfo).setPEnabledExtensionNames(extensions);

	if(enableValidationLayers)
		instanceInfo.setPEnabledLayerNames(validationLayers);

	instance = vk::createInstanceUnique(instanceInfo);
}

bool Multisampling::checkValidationLayerSupport()
{
	auto availableLayers = vk::enumerateInstanceLayerProperties();

	for(const char* layerName : validationLayers)
	{
		bool layerFound = false;
		for(const auto& layerProperties : availableLayers)
		{
			if(strcmp(layerProperties.layerName, layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}
		if(!layerFound)
			return false;
	}
	return true;
}

std::vector<const char*> Multisampling::getRequiredExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if(enableValidationLayers)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

void Multisampling::setupDebugMessenger()
{
	if(!enableValidationLayers)
		return;

	auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT()
						  .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
											  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
						  .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
										  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
										  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
						  .setPfnUserCallback(debugCallback);

	pfnVkCreateDebugUtilsMessengerEXT = reinterpret_cast<decltype(pfnVkCreateDebugUtilsMessengerEXT)>(
		instance->getProcAddr("vkCreateDebugUtilsMessengerEXT"));
	assert(pfnVkCreateDebugUtilsMessengerEXT);

	pfnVkDestroyDebugUtilsMessengerEXT = reinterpret_cast<decltype(pfnVkDestroyDebugUtilsMessengerEXT)>(
		instance->getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
	assert(pfnVkCreateDebugUtilsMessengerEXT);

	debugMessenger = instance->createDebugUtilsMessengerEXTUnique(createInfo);
}

void Multisampling::createSurface()
{
	VkSurfaceKHR _surface;
	glfwCreateWindowSurface(VkInstance(instance.get()), window, nullptr, &_surface);
	surface = vk::UniqueSurfaceKHR(_surface, instance.get());
}

void Multisampling::pickPhysicalDevice()
{
	auto devices = instance->enumeratePhysicalDevices();
	if(devices.empty())
		throw std::runtime_error("failed to find GPUs  with Vulkan support!");

	for(auto& device : devices)
	{
		if(isDeviceSuitable(device))
		{
			physicalDevice = device;
			break;
		}
	}
}

void Multisampling::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

	float queuePriority = 1.0f;

	for(auto queueFamily : std::set{indices.graphicsFamily.value(), indices.presentFamily.value()})
		queueCreateInfos.push_back(
			vk::DeviceQueueCreateInfo().setQueueFamilyIndex(queueFamily).setQueuePriorities(queuePriority));

	auto enabledFeatures = vk::PhysicalDeviceFeatures().setSamplerAnisotropy(true);

	auto createInfo = vk::DeviceCreateInfo()
						  .setQueueCreateInfos(queueCreateInfos)
						  .setPEnabledExtensionNames(deviceExtensions)
						  .setPEnabledFeatures(&enabledFeatures);

	device = physicalDevice.createDeviceUnique(createInfo);

	graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
	presentQueue = device->getQueue(indices.presentFamily.value(), 0);
}

void Multisampling::createFramebuffers()
{
	swapChainFramebuffers.clear();
	swapChainFramebuffers.reserve(swapChainImageViews.size());

	for(auto& imageView : swapChainImageViews)
	{
		std::array attachments = {imageView.get(), depthImageView.get()};

		auto framebufferInfo = vk::FramebufferCreateInfo()
								   .setRenderPass(renderPass.get())
								   .setAttachments(attachments)
								   .setWidth(swapChainExtent.width)
								   .setHeight(swapChainExtent.height)
								   .setLayers(1);
		swapChainFramebuffers.push_back(device->createFramebufferUnique(framebufferInfo));
	}
}

void Multisampling::createCommandPool()
{
	QueueFamilyIndices queueFamilyindices = findQueueFamilies(physicalDevice);
	auto poolInfo = vk::CommandPoolCreateInfo().setQueueFamilyIndex(queueFamilyindices.graphicsFamily.value());
	commandPool = device->createCommandPoolUnique(poolInfo);
}

void Multisampling::createSyncObjects()
{
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
	imagesInFlight.clear();
	imagesInFlight.resize(swapChainImages.size());

	auto fenceInfo = vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled);

	for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		imageAvailableSemaphores[i] = device->createSemaphoreUnique({});
		renderFinishedSemaphores[i] = device->createSemaphoreUnique({});
		inFlightFences[i] = device->createFenceUnique(fenceInfo);
	}
}

void Multisampling::mainLoop()
{
	while(!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		currentFOV += (targetFOV - currentFOV) * 0.005;
		drawFrame();
	}
	device->waitIdle();
}

void Multisampling::cleanup()
{
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool Multisampling::isDeviceSuitable(vk::PhysicalDevice device)
{
	bool swapChainAdequate = false;
	if(checkDeviceExtensionSupport(device))
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}
	auto supportedFeatures = device.getFeatures();

	return findQueueFamilies(device).isComplete() && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

bool Multisampling::checkDeviceExtensionSupport(vk::PhysicalDevice device)
{
	std::vector<vk::ExtensionProperties> availableExtensions;
	availableExtensions = device.enumerateDeviceExtensionProperties();

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
	for(auto extension : availableExtensions)
		requiredExtensions.erase(extension.extensionName);

	return requiredExtensions.empty();
}

void Multisampling::drawFrame()
{
	device->waitForFences(inFlightFences[currentFrame].get(), true, UINT64_MAX);

	try
	{
		uint32_t imageIndex = device
								  ->acquireNextImageKHR(swapChain.get(), UINT64_MAX,
									  imageAvailableSemaphores[currentFrame].get(), nullptr)
								  .value;

		if(imagesInFlight[imageIndex])
			device->waitForFences(imagesInFlight[imageIndex].get(), true, UINT64_MAX);
		imagesInFlight[imageIndex] = device->createFenceUnique({vk::FenceCreateFlagBits::eSignaled});

		updateUniformBuffer(imageIndex);

		vk::PipelineStageFlags waitStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		auto submitInfo = vk::SubmitInfo()
							  .setWaitSemaphores(imageAvailableSemaphores[currentFrame].get())
							  .setSignalSemaphores(renderFinishedSemaphores[currentFrame].get())
							  .setCommandBuffers(commandBuffers[imageIndex].get())
							  .setWaitDstStageMask(waitStageMask);

		device->resetFences(inFlightFences[currentFrame].get());

		graphicsQueue.submit(submitInfo, inFlightFences[currentFrame].get());

		auto presentInfo = vk::PresentInfoKHR()
							   .setWaitSemaphores(renderFinishedSemaphores[currentFrame].get())
							   .setSwapchains(swapChain.get())
							   .setImageIndices(imageIndex);

		auto result = presentQueue.presentKHR(presentInfo);

		if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}
	catch(vk::OutOfDateKHRError& e)
	{
		recreateSwapChain();
	}
}

void Multisampling::recreateSwapChain()
{
	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while(width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	device->waitIdle();

	createSwapChain();
	createImageViews();
	createRenderPass();
	createGraphicsPipeline();
	createDepthResources();
	createFramebuffers();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();
}

void Multisampling::createDescriptorPool()
{
	std::vector<vk::DescriptorPoolSize> poolSizes;
	poolSizes.push_back(vk::DescriptorPoolSize()
							.setType(vk::DescriptorType::eUniformBuffer)
							.setDescriptorCount(swapChainImages.size()));
	poolSizes.push_back(vk::DescriptorPoolSize()
							.setType(vk::DescriptorType::eCombinedImageSampler)
							.setDescriptorCount(swapChainImages.size()));

	auto poolInfo = vk::DescriptorPoolCreateInfo().setPoolSizes(poolSizes).setMaxSets(swapChainImages.size());

	descriptorPool = device->createDescriptorPoolUnique(poolInfo);
}

void Multisampling::createDescriptorSets()
{
	auto layouts = std::vector<vk::DescriptorSetLayout>(swapChainImages.size(), descriptorSetLayout.get());

	auto allocInfo = vk::DescriptorSetAllocateInfo()
						 .setDescriptorPool(descriptorPool.get())
						 .setDescriptorSetCount(swapChainImages.size())
						 .setSetLayouts(layouts);
	descriptorSets = device->allocateDescriptorSets(allocInfo);

	for(int i = 0; i < swapChainImages.size(); i++)
	{
		auto bufferInfo = vk::DescriptorBufferInfo()
							  .setBuffer(uniformBuffers[i].get())
							  .setOffset(0)
							  .setRange(sizeof(UniformBufferObject));

		auto imageInfo = vk::DescriptorImageInfo()
							 .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
							 .setImageView(textureImageView.get())
							 .setSampler(textureSampler.get());

		std::vector<vk::WriteDescriptorSet> descriptorWrites;

		descriptorWrites.push_back(vk::WriteDescriptorSet()
									   .setDstSet(descriptorSets[i])
									   .setDstBinding(0)
									   .setDescriptorType(vk::DescriptorType::eUniformBuffer)
									   .setDescriptorCount(1)
									   .setPBufferInfo(&bufferInfo));

		descriptorWrites.push_back(vk::WriteDescriptorSet()
									   .setDstSet(descriptorSets[i])
									   .setDstBinding(1)
									   .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
									   .setDescriptorCount(1)
									   .setPImageInfo(&imageInfo));

		device->updateDescriptorSets(descriptorWrites, {});
	}
}

void Multisampling::createSwapChain()
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
	auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		imageCount = swapChainSupport.capabilities.maxImageCount;

	auto createInfo = vk::SwapchainCreateInfoKHR()
						  .setSurface(surface.get())
						  .setMinImageCount(imageCount)
						  .setImageFormat(surfaceFormat.format)
						  .setImageColorSpace(surfaceFormat.colorSpace)
						  .setImageExtent(chooseSwapExtent(swapChainSupport.capabilities))
						  .setImageArrayLayers(1)
						  .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
						  .setPreTransform(swapChainSupport.capabilities.currentTransform)
						  .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
						  .setPresentMode(chooseSwapPresentMode(swapChainSupport.presentModes))
						  .setClipped(true);

	auto indices = findQueueFamilies(physicalDevice);

	if(indices.graphicsFamily != indices.presentFamily)
	{
		createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
		createInfo.setQueueFamilyIndices(std::array{indices.graphicsFamily.value(), indices.presentFamily.value()});
	}
	else
	{
		createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
	}

	swapChain = device->createSwapchainKHRUnique(createInfo);

	swapChainImages = device->getSwapchainImagesKHR(swapChain.get());

	swapChainImageFormat = createInfo.imageFormat;
	swapChainExtent = createInfo.imageExtent;
}

vk::Extent2D Multisampling::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
	if(capabilities.currentExtent.width != UINT32_MAX)
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetWindowSize(window, &width, &height);

		vk::Extent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

		actualExtent.width = std::max(
			capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(
			capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}

vk::PresentModeKHR Multisampling::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& avaialablePresentModes)
{
	for(const auto& avaialblePresentMode : avaialablePresentModes)
		if(avaialblePresentMode == vk::PresentModeKHR::eMailbox)
			return avaialblePresentMode;
	return vk::PresentModeKHR::eFifo;
}

vk::SurfaceFormatKHR Multisampling::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for(const auto& availableFormat : availableFormats)
		if(availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
			availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			return availableFormat;

	return availableFormats.front();
}

SwapChainSupportDetails Multisampling::querySwapChainSupport(vk::PhysicalDevice device)
{
	SwapChainSupportDetails details;
	details.capabilities = device.getSurfaceCapabilitiesKHR(surface.get());
	details.formats = device.getSurfaceFormatsKHR(surface.get());
	details.presentModes = device.getSurfacePresentModesKHR(surface.get());
	return details;
}

vk::UniqueImageView Multisampling::createImageView(
	vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
{
	auto createInfo = vk::ImageViewCreateInfo().setImage(image).setViewType(vk::ImageViewType::e2D).setFormat(format);
	createInfo.subresourceRange.setAspectMask(aspectFlags).setLevelCount(mipLevels).setLayerCount(1);
	return device->createImageViewUnique(createInfo);
}

void Multisampling::createImageViews()
{
	swapChainImageViews.resize(swapChainImages.size());
	for(size_t i = 0; i < swapChainImages.size(); i++)
		swapChainImageViews[i] =
			createImageView(swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1);
}

void Multisampling::createDepthResources()
{
	auto format = findDepthFormat();
	std::tie(depthImage, depthImageMemory) =
		createImage(swapChainExtent.width, swapChainExtent.height, 1, format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal);
	depthImageView = createImageView(depthImage.get(), format, vk::ImageAspectFlagBits::eDepth, 1);

	transitionImageLayout(
		depthImage.get(), format, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
}

bool Multisampling::hasStencilComponent(vk::Format format)
{
	return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

vk::Format Multisampling::findDepthFormat()
{
	return findSupportedFormat({vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint, vk::Format::eD32Sfloat},
		vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

vk::Format Multisampling::findSupportedFormat(
	const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
{
	for(auto desiredFormat : candidates)
	{
		auto properties = physicalDevice.getFormatProperties(desiredFormat);
		if(tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features)
		{
			return desiredFormat;
		}
		else if(tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features)
		{
			return desiredFormat;
		}
	}
	throw std::runtime_error("could not find suitable format");
}

void Multisampling::createRenderPass()
{
	auto colorAttachment = vk::AttachmentDescription()
							   .setFormat(swapChainImageFormat)
							   .setSamples(vk::SampleCountFlagBits::e1)
							   .setLoadOp(vk::AttachmentLoadOp::eClear)
							   .setStoreOp(vk::AttachmentStoreOp::eStore)
							   .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
							   .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
							   .setInitialLayout(vk::ImageLayout::eUndefined)
							   .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	auto colorAttachmentRef =
		vk::AttachmentReference().setAttachment(0).setLayout(vk::ImageLayout::eColorAttachmentOptimal);

	auto depthStencilAttachment = vk::AttachmentDescription()
									  .setFormat(findDepthFormat())
									  .setSamples(vk::SampleCountFlagBits::e1)
									  .setLoadOp(vk::AttachmentLoadOp::eClear)
									  .setStoreOp(vk::AttachmentStoreOp::eDontCare)
									  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
									  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
									  .setInitialLayout(vk::ImageLayout::eUndefined)
									  .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	auto depthStencilAttachmentRef =
		vk::AttachmentReference().setAttachment(1).setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	auto subpass = vk::SubpassDescription()
					   .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
					   .setColorAttachments(colorAttachmentRef)
					   .setPDepthStencilAttachment(&depthStencilAttachmentRef);

	auto dependency = vk::SubpassDependency()
						  .setSrcSubpass(VK_SUBPASS_EXTERNAL)
						  .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
						  .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
						  .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

	std::array attachments = {colorAttachment, depthStencilAttachment};

	auto renderPassInfo =
		vk::RenderPassCreateInfo().setAttachments(attachments).setSubpasses(subpass).setDependencies(dependency);

	renderPass = device->createRenderPassUnique(renderPassInfo);
}

void Multisampling::createDescriptorSetLayout()
{
	std::vector<vk::DescriptorSetLayoutBinding> bindings;

	bindings.push_back(vk::DescriptorSetLayoutBinding()
						   .setBinding(0)
						   .setDescriptorType(vk::DescriptorType::eUniformBuffer)
						   .setDescriptorCount(1)
						   .setStageFlags(vk::ShaderStageFlagBits::eVertex));

	bindings.push_back(vk::DescriptorSetLayoutBinding()
						   .setBinding(1)
						   .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
						   .setDescriptorCount(1)
						   .setStageFlags(vk::ShaderStageFlagBits::eFragment));

	auto layoutInfo = vk::DescriptorSetLayoutCreateInfo().setBindings(bindings);

	descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutInfo);
}

void Multisampling::createGraphicsPipeline()
{
	auto vertShaderCode = readFile("shader.vert.spv");
	auto fragShaderCode = readFile("shader.frag.spv");

	auto vertShaderModule = createShaderModule(vertShaderCode);
	auto fragShaderModule = createShaderModule(fragShaderCode);

	auto vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
								   .setStage(vk::ShaderStageFlagBits::eVertex)
								   .setModule(vertShaderModule.get())
								   .setPName("main");

	auto fragShaderStageInfo = vk::PipelineShaderStageCreateInfo()
								   .setStage(vk::ShaderStageFlagBits::eFragment)
								   .setModule(fragShaderModule.get())
								   .setPName("main");

	auto vertexBindingDescription = Vertex::getBindingDescription();
	auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();

	auto vertexInputState = vk::PipelineVertexInputStateCreateInfo()
								.setVertexBindingDescriptions(vertexBindingDescription)
								.setVertexAttributeDescriptions(vertexAttributeDescriptions);

	auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo().setTopology(vk::PrimitiveTopology::eTriangleList);

	auto viewport = vk::Viewport()
						.setWidth(swapChainExtent.width)
						.setHeight(swapChainExtent.height)
						.setMinDepth(0.0f)
						.setMaxDepth(1.0f);

	auto scissor = vk::Rect2D().setExtent(swapChainExtent);

	auto viewportState = vk::PipelineViewportStateCreateInfo().setViewports(viewport).setScissors(scissor);

	auto rasterizer = vk::PipelineRasterizationStateCreateInfo()
						  .setPolygonMode(vk::PolygonMode::eFill)
						  .setLineWidth(1.0f)
						  .setCullMode(vk::CullModeFlagBits::eNone)
						  .setFrontFace(vk::FrontFace::eCounterClockwise);

	auto multisampling = vk::PipelineMultisampleStateCreateInfo()
							 .setRasterizationSamples(vk::SampleCountFlagBits::e1)
							 .setMinSampleShading(1.0f);

	auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState().setColorWriteMask(
		vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA);

	auto colorBlending = vk::PipelineColorBlendStateCreateInfo().setAttachments(colorBlendAttachment);

	auto depthStencil = vk::PipelineDepthStencilStateCreateInfo()
							.setDepthTestEnable(true)
							.setDepthCompareOp(vk::CompareOp::eLess)
							.setDepthWriteEnable(true);

	auto pipeLineLayoutInfo = vk::PipelineLayoutCreateInfo().setSetLayouts(descriptorSetLayout.get());

	pipelineLayout = device->createPipelineLayoutUnique(pipeLineLayoutInfo);

	auto stages = std::array{vertShaderStageInfo, fragShaderStageInfo};

	auto pipelineInfo = vk::GraphicsPipelineCreateInfo()
							.setStages(stages)
							.setPInputAssemblyState(&inputAssembly)
							.setPVertexInputState(&vertexInputState)
							.setPViewportState(&viewportState)
							.setPRasterizationState(&rasterizer)
							.setPMultisampleState(&multisampling)
							.setPColorBlendState(&colorBlending)
							.setPDepthStencilState(&depthStencil)
							.setLayout(pipelineLayout.get())
							.setRenderPass(renderPass.get());

	graphicsPipeline = device->createGraphicsPipelineUnique(nullptr, pipelineInfo).value;
}

void Multisampling::updateUniformBuffer(uint32_t currentImage)
{
	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - appStartTime).count();

	UniformBufferObject ubo;
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	float aspectRatio = swapChainExtent.width / float(swapChainExtent.height);
	ubo.proj = glm::perspective(glm::radians(currentFOV), aspectRatio, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;

	void* data = device->mapMemory(uniformBuffersMemory[currentImage].get(), 0, sizeof(ubo));
	std::memcpy(data, &ubo, sizeof(ubo));
	device->unmapMemory(uniformBuffersMemory[currentImage].get());
}

vk::UniqueShaderModule Multisampling::createShaderModule(const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	return device->createShaderModuleUnique(createInfo);
}

vk::UniqueCommandBuffer Multisampling::beginSingleTimeCommands()
{
	auto allocInfo = vk::CommandBufferAllocateInfo()
						 .setCommandPool(commandPool.get())
						 .setLevel(vk::CommandBufferLevel::ePrimary)
						 .setCommandBufferCount(1);

	auto commandBuffers = device->allocateCommandBuffersUnique(allocInfo);

	commandBuffers[0]->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	return std::move(commandBuffers[0]);
}

void Multisampling::endSingleTimeCommands(vk::UniqueCommandBuffer&& commandBuffer)
{
	commandBuffer->end();
	graphicsQueue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer.get()), nullptr);
	graphicsQueue.waitIdle();
}

void Multisampling::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
	auto copyCommandBuffer = beginSingleTimeCommands();

	copyCommandBuffer->copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy{0, 0, size});

	endSingleTimeCommands(std::move(copyCommandBuffer));
}

void Multisampling::copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, uint32_t width, uint32_t height)
{
	auto copyCommandBuffer = beginSingleTimeCommands();

	auto imageCopyInfo =
		vk::BufferImageCopy().setImageExtent(vk::Extent3D().setWidth(width).setHeight(height).setDepth(1));

	imageCopyInfo.imageSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor).setLayerCount(1);

	copyCommandBuffer->copyBufferToImage(srcBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, imageCopyInfo);

	endSingleTimeCommands(std::move(copyCommandBuffer));
}

void Multisampling::transitionImageLayout(
	vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
{
	auto commandBuffer = beginSingleTimeCommands();

	auto barrier = vk::ImageMemoryBarrier()
					   .setOldLayout(oldLayout)
					   .setNewLayout(newLayout)
					   .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
					   .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
					   .setImage(image);
	barrier.subresourceRange.setLevelCount(mipLevels).setLayerCount(1);

	if(newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
	{
		barrier.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eDepth);
		if(hasStencilComponent(format))
			barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
	}
	else
	{
		barrier.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
	}

	vk::PipelineStageFlags sourceStageMask;
	vk::PipelineStageFlags destinationStageMask;

	if(oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
	{
		barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
		sourceStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStageMask = vk::PipelineStageFlagBits::eTransfer;
	}
	else if(oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
	{
		barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
		barrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
		sourceStageMask = vk::PipelineStageFlagBits::eTransfer;
		destinationStageMask = vk::PipelineStageFlagBits::eFragmentShader;
	}
	else if(oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
	{
		barrier.setDstAccessMask(
			vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite);
		sourceStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
		destinationStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests;
	}
	else
	{
		throw std::runtime_error("unsupported layout transition!");
	}
	commandBuffer->pipelineBarrier(sourceStageMask, destinationStageMask, {}, {}, {}, barrier);

	endSingleTimeCommands(std::move(commandBuffer));
}

std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory> Multisampling::createBuffer(
	vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
{
	auto bufferInfo = vk::BufferCreateInfo().setSize(size).setUsage(usage).setSharingMode(vk::SharingMode::eExclusive);

	auto buffer = device->createBufferUnique(bufferInfo);

	auto memRequirements = device->getBufferMemoryRequirements(buffer.get());

	auto allocInfo = vk::MemoryAllocateInfo()
						 .setAllocationSize(memRequirements.size)
						 .setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));

	auto memory = device->allocateMemoryUnique(allocInfo);

	device->bindBufferMemory(buffer.get(), memory.get(), 0);

	return {std::move(buffer), std::move(memory)};
}

std::pair<vk::UniqueImage, vk::UniqueDeviceMemory> Multisampling::createImage(uint32_t width, uint32_t height,
	uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
	vk::MemoryPropertyFlags properties)
{
	auto imageInfo = vk::ImageCreateInfo()
						 .setImageType(vk::ImageType::e2D)
						 .setMipLevels(mipLevels)
						 .setArrayLayers(1)
						 .setFormat(format)
						 .setTiling(tiling)
						 .setUsage(usage)
						 .setSharingMode(vk::SharingMode::eExclusive)
						 .setSamples(vk::SampleCountFlagBits::e1);
	imageInfo.extent.setWidth(width).setHeight(height).setDepth(1);

	auto image = device->createImageUnique(imageInfo);

	auto memoryRequirements = device->getImageMemoryRequirements(image.get());
	auto allocInfo = vk::MemoryAllocateInfo()
						 .setAllocationSize(memoryRequirements.size)
						 .setMemoryTypeIndex(findMemoryType(memoryRequirements.memoryTypeBits, properties));
	auto imageMemory = device->allocateMemoryUnique(allocInfo);
	device->bindImageMemory(image.get(), imageMemory.get(), 0);
	return {std::move(image), std::move(imageMemory)};
}

void Multisampling::loadModel()
{
	tinyobj::attrib_t attribute;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string error;

	if(!tinyobj::LoadObj(&attribute, &shapes, &materials, &error, meshPath.c_str()))
	{
		throw std::runtime_error(error);
	}

	std::unordered_map<Vertex, uint32_t> uniqueVertices;

	for(const auto& shape : shapes)
	{
		for(const auto& index : shape.mesh.indices)
		{
			Vertex vertex;

			vertex.pos = {attribute.vertices[3 * index.vertex_index + 0],
				attribute.vertices[3 * index.vertex_index + 1], attribute.vertices[3 * index.vertex_index + 2]};

			vertex.texCoord = {attribute.texcoords[2 * index.texcoord_index + 0],
				1.0f - attribute.texcoords[2 * index.texcoord_index + 1]};

			vertex.color = {1.0f, 1.0f, 1.0f};

			if(uniqueVertices.count(vertex) == 0)
			{
				uniqueVertices[vertex] = vertices.size();
				vertices.push_back(vertex);
			}

			indices.push_back(uniqueVertices[vertex]);
		}
	}
}

void Multisampling::createVertexBuffer()
{
	auto size = sizeof(vertices[0]) * vertices.size();

	auto [stagingBuffer, stagingBufferMemory] = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

	void* data = device->mapMemory(stagingBufferMemory.get(), 0, size);
	std::memcpy(data, vertices.data(), size);
	device->unmapMemory(stagingBufferMemory.get());

	std::tie(vertexBuffer, vertexBufferMemory) =
		createBuffer(size, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

	copyBuffer(stagingBuffer.get(), vertexBuffer.get(), size);
}

void Multisampling::generateMipmaps(
	vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
{
	auto formatProperties = physicalDevice.getFormatProperties(imageFormat);
	if(!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
		throw std::runtime_error("texture image format does not support linear blitting!");

	auto commandBuffer = beginSingleTimeCommands();

	auto barrier = vk::ImageMemoryBarrier()
					   .setImage(image)
					   .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
					   .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
	barrier.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor)
		.setBaseArrayLayer(0)
		.setLayerCount(1)
		.setLevelCount(1);

	uint32_t mipWidth = texWidth;
	uint32_t mipHeight = texHeight;

	for(uint32_t i = 1; i < mipLevels; i++)
	{
		barrier.subresourceRange.setBaseMipLevel(i - 1);
		barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
			.setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
			.setDstAccessMask(vk::AccessFlagBits::eTransferRead);

		commandBuffer->pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

		auto blit = vk::ImageBlit();
		blit.srcOffsets[0] = vk::Offset3D(0, 0, 0);
		blit.srcOffsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);
		blit.srcSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
			.setMipLevel(i - 1)
			.setBaseArrayLayer(0)
			.setLayerCount(1);

		blit.dstOffsets[0] = vk::Offset3D(0, 0, 0);
		blit.dstOffsets[1] = vk::Offset3D(std::max(mipWidth / 2, 1u), std::max(mipHeight / 2, 1u), 1);
		blit.dstSubresource.setAspectMask(vk::ImageAspectFlagBits::eColor)
			.setMipLevel(i)
			.setBaseArrayLayer(0)
			.setLayerCount(1);

		commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image,
			vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

		barrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
			.setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
			.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

		commandBuffer->pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

		if(mipWidth > 1)
			mipWidth /= 2;
		if(mipHeight > 1)
			mipHeight /= 2;
	}

	barrier.subresourceRange.setBaseMipLevel(mipLevels - 1);
	barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
		.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
		.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
		.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
	commandBuffer->pipelineBarrier(
		vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

	endSingleTimeCommands(std::move(commandBuffer));
}

void Multisampling::createTextureImage()
{
	int texWidth, texHeight, texChannels;
	auto pixels = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	if(!pixels)
		throw std::runtime_error("failed to load texture image");

	auto imageSize = texWidth * texHeight * 4;

	mipLevels = std::floor(std::log2f(std::max(texWidth, texHeight))) + 1;

	auto [stagingBuffer, stagingBufferMemory] = createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	void* data = device->mapMemory(stagingBufferMemory.get(), 0, imageSize);
	std::memcpy(data, pixels, imageSize);
	device->unmapMemory(stagingBufferMemory.get());
	stbi_image_free(pixels);

	std::tie(textureImage, textureImageMemory) = createImage(texWidth, texHeight, mipLevels, vk::Format::eR8G8B8A8Srgb,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal);

	transitionImageLayout(textureImage.get(), vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined,
		vk::ImageLayout::eTransferDstOptimal, mipLevels);

	copyBufferToImage(stagingBuffer.get(), textureImage.get(), texWidth, texHeight);

	generateMipmaps(textureImage.get(), vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
}

void Multisampling::createTextureImageView()
{
	textureImageView =
		createImageView(textureImage.get(), vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
}

void Multisampling::createTextureSampler()
{
	auto samplerInfo = vk::SamplerCreateInfo()
						   .setMagFilter(vk::Filter::eLinear)
						   .setMinFilter(vk::Filter::eLinear)
						   .setAddressModeU(vk::SamplerAddressMode::eRepeat)
						   .setAddressModeV(vk::SamplerAddressMode::eRepeat)
						   .setAddressModeW(vk::SamplerAddressMode::eRepeat)
						   .setAnisotropyEnable(true)
						   .setMaxAnisotropy(16.0f)
						   .setMinLod(0)
						   .setMaxLod(mipLevels)
						   .setMipmapMode(vk::SamplerMipmapMode::eLinear)
						   .setMipLodBias(0);
	textureSampler = device->createSamplerUnique(samplerInfo);
}

void Multisampling::createIndexBuffer()
{
	auto size = sizeof(indices[0]) * indices.size();

	auto [stagingBuffer, stagingBufferMemory] = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

	void* data = device->mapMemory(stagingBufferMemory.get(), 0, size);
	std::memcpy(data, indices.data(), size);
	device->unmapMemory(stagingBufferMemory.get());

	std::tie(indexBuffer, indexBufferMemory) =
		createBuffer(size, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal);

	copyBuffer(stagingBuffer.get(), indexBuffer.get(), size);
}

void Multisampling::createUniformBuffers()
{
	auto size = sizeof(UniformBufferObject);
	uniformBuffers.resize(swapChainImages.size());
	uniformBuffersMemory.resize(swapChainImages.size());
	for(int i = 0; i < swapChainImages.size(); i++)
	{
		std::tie(uniformBuffers[i], uniformBuffersMemory[i]) =
			createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	}
}

void Multisampling::createCommandBuffers()
{
	auto allocInfo = vk::CommandBufferAllocateInfo()
						 .setCommandPool(commandPool.get())
						 .setLevel(vk::CommandBufferLevel::ePrimary)
						 .setCommandBufferCount(swapChainFramebuffers.size());

	commandBuffers = device->allocateCommandBuffersUnique(allocInfo);

	for(int i = 0; i < commandBuffers.size(); i++)
	{
		auto beginInfo = vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

		commandBuffers[i]->begin(beginInfo);

		std::array<vk::ClearValue, 2> clearValues = {
			vk::ClearColorValue().setFloat32({0.0f, 0.0f, 0.0f, 1.0f}), vk::ClearDepthStencilValue().setDepth(1.0f)};

		auto renderPassInfo = vk::RenderPassBeginInfo()
								  .setRenderPass(renderPass.get())
								  .setFramebuffer(swapChainFramebuffers[i].get())
								  .setClearValues(clearValues);

		renderPassInfo.renderArea.extent = swapChainExtent;

		commandBuffers[i]->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.get());
		commandBuffers[i]->bindVertexBuffers(0, vertexBuffer.get(), vk::DeviceSize(0));
		commandBuffers[i]->bindIndexBuffer(indexBuffer.get(), vk::DeviceSize(0), vk::IndexType::eUint32);
		commandBuffers[i]->bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, descriptorSets[i], {});
		commandBuffers[i]->drawIndexed(indices.size(), 1, 0, 0, 0);
		commandBuffers[i]->endRenderPass();
		commandBuffers[i]->end();
	}
}

uint32_t Multisampling::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	auto memProperties = physicalDevice.getMemoryProperties();
	for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		if(typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	throw std::runtime_error("No suitable memory type found!");
}

QueueFamilyIndices Multisampling::findQueueFamilies(vk::PhysicalDevice device)
{
	QueueFamilyIndices indices;
	std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
	int i = 0;
	for(auto const& queueFamily : queueFamilies)
	{
		if(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			indices.graphicsFamily = i;

		if(device.getSurfaceSupportKHR(i, surface.get()))
			indices.presentFamily = i;
		if(indices.isComplete())
			break;
		i++;
	}
	return indices;
}