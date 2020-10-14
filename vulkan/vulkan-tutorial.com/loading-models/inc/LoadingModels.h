#pragma once

#include <vulkan/vulkan.hpp>

#define GLM_ENABLE_EXPERIMENTAL

#include <GLFW/glfw3.h>
#include <chrono>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <optional>
#include <vector>

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return vk::VertexInputBindingDescription()
			.setBinding(0)
			.setStride(sizeof(Vertex))
			.setInputRate(vk::VertexInputRate::eVertex);
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		auto positionAttribute = vk::VertexInputAttributeDescription()
									 .setBinding(0)
									 .setLocation(0)
									 .setFormat(vk::Format::eR32G32B32Sfloat)
									 .setOffset(offsetof(Vertex, pos));

		auto colorAttribute = vk::VertexInputAttributeDescription()
								  .setBinding(0)
								  .setLocation(1)
								  .setFormat(vk::Format::eR32G32B32Sfloat)
								  .setOffset(offsetof(Vertex, color));

		auto texCoordAttribute = vk::VertexInputAttributeDescription()
									 .setBinding(0)
									 .setLocation(2)
									 .setFormat(vk::Format::eR32G32Sfloat)
									 .setOffset(offsetof(Vertex, texCoord));

		return {positionAttribute, colorAttribute, texCoordAttribute};
	}

	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std
{
template <>
struct hash<Vertex>
{
	std::size_t operator()(const Vertex& key) const
	{
		return ((hash<glm::vec3>()(key.pos) ^ (hash<glm::vec3>()(key.color) << 1)) >> 1) ^
			   (hash<glm::vec2>()(key.texCoord) << 1);
	}
};
} // namespace std

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

class LoadingModels
{
public:
	bool framebufferResized = false;
	void run();

private:
	const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
	const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	const int WIDTH = 800;
	const int HEIGHT = 800;
	const int MAX_FRAMES_IN_FLIGHT = 2;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	std::chrono::high_resolution_clock::time_point appStartTime = std::chrono::high_resolution_clock::now();
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
	vk::UniqueImage depthImage;
	vk::UniqueDeviceMemory depthImageMemory;
	vk::UniqueImageView depthImageView;
	vk::UniqueRenderPass renderPass;
	vk::UniqueDescriptorSetLayout descriptorSetLayout;
	vk::UniquePipelineLayout pipelineLayout;
	vk::UniquePipeline graphicsPipeline;
	std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;
	vk::UniqueCommandPool commandPool;
	std::vector<vk::UniqueCommandBuffer> commandBuffers;
	vk::UniqueBuffer vertexBuffer;
	vk::UniqueDeviceMemory vertexBufferMemory;
	vk::UniqueBuffer indexBuffer;
	vk::UniqueDeviceMemory indexBufferMemory;
	vk::UniqueImage textureImage;
	vk::UniqueDeviceMemory textureImageMemory;
	vk::UniqueImageView textureImageView;
	vk::UniqueSampler textureSampler;
	vk::UniqueDescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets;
	std::vector<vk::UniqueBuffer> uniformBuffers;
	std::vector<vk::UniqueDeviceMemory> uniformBuffersMemory;
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
	void createDescriptorPool();
	void createDescriptorSets();
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
	vk::UniqueImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags);
	void createImageViews();
	void createDepthResources();
	bool hasStencilComponent(vk::Format format);
	vk::Format findDepthFormat();
	vk::Format findSupportedFormat(
		const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
	void createRenderPass();
	void createDescriptorSetLayout();
	void createGraphicsPipeline();
	void updateUniformBuffer(uint32_t currentImage);
	vk::UniqueShaderModule createShaderModule(const std::vector<char>& code);
	vk::UniqueCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(vk::UniqueCommandBuffer&& commandBuffer);
	void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
	void copyBufferToImage(vk::Buffer srcBuffer, vk::Image dstImage, uint32_t width, uint32_t height);
	void transitionImageLayout(
		vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
	std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory> createBuffer(
		vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
	std::pair<vk::UniqueImage, vk::UniqueDeviceMemory> createImage(uint32_t width, uint32_t height, vk::Format format,
		vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties);
	void loadModel();
	void createVertexBuffer();
	void createTextureImage();
	void createTextureImageView();
	void createTextureSampler();
	void createIndexBuffer();
	void createUniformBuffers();
	void createCommandBuffers();
	QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
	void cleanup();
};