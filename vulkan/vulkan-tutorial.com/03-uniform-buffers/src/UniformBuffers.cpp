#include "UniformBuffers.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <set>
#include <stdexcept>

#ifdef DEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif

const std::vector<Vertex> vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}, {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

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
	UniformBuffers* app = reinterpret_cast<UniformBuffers*>(glfwGetWindowUserPointer(window));
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
	std::cerr << "validation layer:  " << pCallbackData->pMessage << '\n';

	return false;
}

void UniformBuffers::run()
{
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}

void UniformBuffers::initWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

	glfwSetWindowUserPointer(window, this);

	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	glfwSetKeyCallback(window, keypressCallback);
}

void UniformBuffers::initVulkan()
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
	createFramebuffers();
	createCommandPool();
	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();
	createSyncObjects();
}

void UniformBuffers::createInstance()
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

bool UniformBuffers::checkValidationLayerSupport()
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

std::vector<const char*> UniformBuffers::getRequiredExtensions()
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

void UniformBuffers::setupDebugMessenger()
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

void UniformBuffers::createSurface()
{
	VkSurfaceKHR _surface;
	glfwCreateWindowSurface(VkInstance(instance.get()), window, nullptr, &_surface);
	surface = vk::UniqueSurfaceKHR(_surface, instance.get());
}

void UniformBuffers::pickPhysicalDevice()
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

void UniformBuffers::createLogicalDevice()
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

	float queuePriority = 1.0f;

	for(auto queueFamily : std::set{indices.graphicsFamily.value(), indices.presentFamily.value()})
		queueCreateInfos.push_back(
			vk::DeviceQueueCreateInfo().setQueueFamilyIndex(queueFamily).setQueuePriorities(queuePriority));

	auto createInfo =
		vk::DeviceCreateInfo().setQueueCreateInfos(queueCreateInfos).setPEnabledExtensionNames(deviceExtensions);

	device = physicalDevice.createDeviceUnique(createInfo);

	graphicsQueue = device->getQueue(indices.graphicsFamily.value(), 0);
	presentQueue = device->getQueue(indices.presentFamily.value(), 0);
}

void UniformBuffers::createFramebuffers()
{
	swapChainFramebuffers.clear();
	swapChainFramebuffers.reserve(swapChainImageViews.size());

	for(auto& imageView : swapChainImageViews)
	{
		auto framebufferInfo = vk::FramebufferCreateInfo()
								   .setRenderPass(renderPass.get())
								   .setAttachments(imageView.get())
								   .setWidth(swapChainExtent.width)
								   .setHeight(swapChainExtent.height)
								   .setLayers(1);
		swapChainFramebuffers.push_back(device->createFramebufferUnique(framebufferInfo));
	}
}

void UniformBuffers::createCommandPool()
{
	QueueFamilyIndices queueFamilyindices = findQueueFamilies(physicalDevice);
	auto poolInfo = vk::CommandPoolCreateInfo().setQueueFamilyIndex(queueFamilyindices.graphicsFamily.value());
	commandPool = device->createCommandPoolUnique(poolInfo);
}

void UniformBuffers::createSyncObjects()
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

void UniformBuffers::mainLoop()
{
	while(!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		drawFrame();
	}
	device->waitIdle();
}

void UniformBuffers::cleanup()
{
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool UniformBuffers::isDeviceSuitable(vk::PhysicalDevice device)
{
	bool swapChainAdequate = false;
	if(checkDeviceExtensionSupport(device))
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}
	return findQueueFamilies(device).isComplete() && swapChainAdequate;
}

bool UniformBuffers::checkDeviceExtensionSupport(vk::PhysicalDevice device)
{
	std::vector<vk::ExtensionProperties> availableExtensions;
	availableExtensions = device.enumerateDeviceExtensionProperties();

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
	for(auto extension : availableExtensions)
		requiredExtensions.erase(extension.extensionName);

	return requiredExtensions.empty();
}

void UniformBuffers::drawFrame()
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

void UniformBuffers::recreateSwapChain()
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
	createFramebuffers();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();
}

void UniformBuffers::createDescriptorPool()
{
	auto poolSize =
		vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(swapChainImages.size());

	auto poolInfo = vk::DescriptorPoolCreateInfo().setPoolSizes(poolSize).setMaxSets(swapChainImages.size());

	descriptorPool = device->createDescriptorPoolUnique(poolInfo);
}

void UniformBuffers::createDescriptorSets()
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

		auto descriptorWrite = vk::WriteDescriptorSet()
								   .setDstSet(descriptorSets[i])
								   .setDstBinding(0)
								   .setDescriptorType(vk::DescriptorType::eUniformBuffer)
								   .setDescriptorCount(1)
								   .setPBufferInfo(&bufferInfo);

		device->updateDescriptorSets(descriptorWrite, {});
	}
}

void UniformBuffers::createSwapChain()
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

vk::Extent2D UniformBuffers::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
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

vk::PresentModeKHR UniformBuffers::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& avaialablePresentModes)
{
	for(const auto& avaialblePresentMode : avaialablePresentModes)
		if(avaialblePresentMode == vk::PresentModeKHR::eMailbox)
			return avaialblePresentMode;
	return vk::PresentModeKHR::eFifo;
}

vk::SurfaceFormatKHR UniformBuffers::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for(const auto& availableFormat : availableFormats)
		if(availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
			availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			return availableFormat;

	return availableFormats.front();
}

SwapChainSupportDetails UniformBuffers::querySwapChainSupport(vk::PhysicalDevice device)
{
	SwapChainSupportDetails details;
	details.capabilities = device.getSurfaceCapabilitiesKHR(surface.get());
	details.formats = device.getSurfaceFormatsKHR(surface.get());
	details.presentModes = device.getSurfacePresentModesKHR(surface.get());
	return details;
}

void UniformBuffers::createImageViews()
{
	swapChainImageViews.resize(swapChainImages.size());

	for(size_t i = 0; i < swapChainImages.size(); i++)
	{
		auto createInfo = vk::ImageViewCreateInfo()
							  .setImage(swapChainImages[i])
							  .setViewType(vk::ImageViewType::e2D)
							  .setFormat(swapChainImageFormat)
							  .setSubresourceRange(vk::ImageSubresourceRange()
													   .setAspectMask(vk::ImageAspectFlagBits::eColor)
													   .setLevelCount(1)
													   .setLayerCount(1)); // horrible?
		swapChainImageViews[i] = device->createImageViewUnique(createInfo);
	}
}

void UniformBuffers::createRenderPass()
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

	auto colorAttachmentRef = vk::AttachmentReference().setLayout(vk::ImageLayout::eColorAttachmentOptimal);

	auto subpass = vk::SubpassDescription()
					   .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
					   .setColorAttachments(colorAttachmentRef);

	auto dependency = vk::SubpassDependency()
						  .setSrcSubpass(VK_SUBPASS_EXTERNAL)
						  .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
						  .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
						  .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

	auto renderPassInfo =
		vk::RenderPassCreateInfo().setAttachments(colorAttachment).setSubpasses(subpass).setDependencies(dependency);

	renderPass = device->createRenderPassUnique(renderPassInfo);
}

void UniformBuffers::createDescriptorSetLayout()
{
	auto uboLayoutBinding = vk::DescriptorSetLayoutBinding()
								.setBinding(0)
								.setDescriptorType(vk::DescriptorType::eUniformBuffer)
								.setDescriptorCount(1)
								.setStageFlags(vk::ShaderStageFlagBits::eVertex);

	auto layoutInfo = vk::DescriptorSetLayoutCreateInfo().setBindings(uboLayoutBinding);

	descriptorSetLayout = device->createDescriptorSetLayoutUnique(layoutInfo);
}

void UniformBuffers::createGraphicsPipeline()
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
						  .setCullMode(vk::CullModeFlagBits::eBack)
						  .setFrontFace(vk::FrontFace::eCounterClockwise);

	auto multisampling = vk::PipelineMultisampleStateCreateInfo()
							 .setRasterizationSamples(vk::SampleCountFlagBits::e1)
							 .setMinSampleShading(1.0f);

	auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState().setColorWriteMask(
		vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA);

	auto colorBlending = vk::PipelineColorBlendStateCreateInfo().setAttachments(colorBlendAttachment);

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
							.setLayout(pipelineLayout.get())
							.setRenderPass(renderPass.get());

	graphicsPipeline = device->createGraphicsPipelineUnique(nullptr, pipelineInfo).value;
}

void UniformBuffers::updateUniformBuffer(uint32_t currentImage)
{
	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - appStartTime).count();

	UniformBufferObject ubo;
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	float aspectRatio = swapChainExtent.width / float(swapChainExtent.height);
	ubo.proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;

	void* data = device->mapMemory(uniformBuffersMemory[currentImage].get(), 0, sizeof(ubo));
	std::memcpy(data, &ubo, sizeof(ubo));
	device->unmapMemory(uniformBuffersMemory[currentImage].get());
}

vk::UniqueShaderModule UniformBuffers::createShaderModule(const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	return device->createShaderModuleUnique(createInfo);
}

void UniformBuffers::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
	auto allocInfo = vk::CommandBufferAllocateInfo()
						 .setCommandPool(commandPool.get())
						 .setLevel(vk::CommandBufferLevel::ePrimary)
						 .setCommandBufferCount(1);

	auto copyCommandBuffers = device->allocateCommandBuffersUnique(allocInfo);

	copyCommandBuffers[0]->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	copyCommandBuffers[0]->copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy{0, 0, size});
	copyCommandBuffers[0]->end();

	graphicsQueue.submit(vk::SubmitInfo().setCommandBuffers(copyCommandBuffers[0].get()), nullptr);

	graphicsQueue.waitIdle();
}

std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory> UniformBuffers::createBuffer(
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

void UniformBuffers::createVertexBuffer()
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

void UniformBuffers::createIndexBuffer()
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

void UniformBuffers::createUniformBuffers()
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

void UniformBuffers::createCommandBuffers()
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

		vk::ClearValue clearColor = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};

		auto renderPassInfo = vk::RenderPassBeginInfo()
								  .setRenderPass(renderPass.get())
								  .setFramebuffer(swapChainFramebuffers[i].get())
								  .setClearValues(clearColor);

		renderPassInfo.renderArea.extent = swapChainExtent;

		commandBuffers[i]->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.get());
		commandBuffers[i]->bindVertexBuffers(0, vertexBuffer.get(), vk::DeviceSize(0));
		commandBuffers[i]->bindIndexBuffer(indexBuffer.get(), vk::DeviceSize(0), vk::IndexType::eUint16);
		commandBuffers[i]->bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, pipelineLayout.get(), 0, descriptorSets[i], {});
		commandBuffers[i]->drawIndexed(indices.size(), 1, 0, 0, 0);
		commandBuffers[i]->endRenderPass();
		commandBuffers[i]->end();
	}
}

uint32_t UniformBuffers::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	auto memProperties = physicalDevice.getMemoryProperties();
	for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		if(typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	throw std::runtime_error("No suitable memory type found!");
}

QueueFamilyIndices UniformBuffers::findQueueFamilies(vk::PhysicalDevice device)
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
