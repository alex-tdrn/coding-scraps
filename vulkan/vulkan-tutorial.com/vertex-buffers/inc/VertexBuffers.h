#pragma once

#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <optional>
#include <vector>

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		vk::VertexInputBindingDescription description;
		description.binding = 0;
		description.stride = sizeof(Vertex);
		description.inputRate = vk::VertexInputRate::eVertex;

		return description;
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		vk::VertexInputAttributeDescription positionAttribute;
		positionAttribute.binding = 0;
		positionAttribute.location = 0;
		positionAttribute.format = vk::Format::eR32G32Sfloat;
		positionAttribute.offset = offsetof(Vertex, pos);

		vk::VertexInputAttributeDescription colorAttribute;
		colorAttribute.binding = 0;
		colorAttribute.location = 1;
		colorAttribute.format = vk::Format::eR32G32B32Sfloat;
		colorAttribute.offset = offsetof(Vertex, color);

		return {positionAttribute, colorAttribute};
	}
};

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	bool isComplete()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

class VertexBuffers
{
public:
	bool framebufferResized = false;
	void run();

private:
	const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
	const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	const int WIDTH = 800;
	const int HEIGHT = 600;
	const int MAX_FRAMES_IN_FLIGHT = 2;

	GLFWwindow* window;
	vk::UniqueInstance instance;
	vk::UniqueDebugUtilsMessengerEXT debugMessenger;
	vk::UniqueSurfaceKHR surface;
	vk::PhysicalDevice physicalDevice;
	vk::UniqueDevice device;
	vk::Queue graphicsQueue;
	vk::Queue presentQueue;
	vk::UniqueSwapchainKHR swapChain;
	std::vector<vk::Image> swapChainImages;
	vk::Format swapChainImageFormat;
	vk::Extent2D swapChainExtent;
	std::vector<vk::UniqueImageView> swapChainImageViews;
	vk::UniqueRenderPass renderPass;
	vk::UniquePipelineLayout pipelineLayout;
	vk::UniquePipeline graphicsPipeline;
	std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;
	vk::UniqueCommandPool commandPool;
	std::vector<vk::UniqueCommandBuffer> commandBuffers;
	std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
	std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
	std::vector<vk::UniqueFence> inFlightFences;
	std::vector<vk::UniqueFence> imagesInFlight;
	size_t currentFrame = 0;

	void initWindow();
	void initVulkan();
	void createInstance();
	bool checkValidationLayerSupport();
	std::vector<const char*> getRequiredExtensions();
	void setupDebugMessenger();
	void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo);
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createFramebuffers();
	void createCommandPool();
	void createSyncObjects();
	void mainLoop();
	bool isDeviceSuitable(vk::PhysicalDevice device);
	bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
	void drawFrame();
	void recreateSwapChain();
	void createSwapChain();
	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& avaialablePresentModes);
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
	SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
	void createImageViews();
	void createRenderPass();
	void createGraphicsPipeline();
	vk::ShaderModule createShaderModule(const std::vector<char>& code);
	void createCommandBuffers();
	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
	void cleanup();
};