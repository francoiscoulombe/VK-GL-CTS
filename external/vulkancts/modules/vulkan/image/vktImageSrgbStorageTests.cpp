/*------------------------------------------------------------------------
 * Vulkan Conformance Tests
 * ------------------------
 *
 * Copyright (c) 2026 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *//*!
 * \file
 * \brief Tests that imageStore through a UNORM view of an sRGB image does
 *        NOT apply sRGB conversion.
 *
 * Rationale:
 *   - SPIR-V has no sRGB image format qualifier (Rgba8 maps to UNORM)
 *   - sRGB formats lack VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
 *   - The Vulkan spec ties sRGB conversion to the *view* format, not the
 *     underlying image format
 *   - When using VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT to create a UNORM view
 *     of an sRGB image for storage, writes must be raw UNORM — no sRGB
 *     conversion
 *
 * Uses distinctive colors (orange, Rebecca purple) so that sRGB leakage
 * is immediately obvious both numerically and visually.
 *
 * This test catches implementations (e.g. Metal-based) that apply sRGB
 * conversion based on the underlying image format rather than the view
 * format.
 *//*--------------------------------------------------------------------*/

#include "vktImageSrgbStorageTests.hpp"
#include "vktImageTestsUtil.hpp"
#include "vktTestCaseUtil.hpp"

#include "vkBuilderUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkImageUtil.hpp"
#include "vkObjUtil.hpp"
#include "vkQueryUtil.hpp"
#include "vkRef.hpp"

#include "deUniquePtr.hpp"
#include "tcuTestLog.hpp"
#include "tcuTextureUtil.hpp"

#include <string>
#include <vector>
#include <cmath>

using namespace vk;

namespace vkt
{
namespace image
{

namespace
{

struct TestParams
{
	VkFormat	srgbFormat;		// The sRGB image creation format
	VkFormat	unormFormat;	// The UNORM view format for storage
};

// Test colors chosen for maximum sRGB/UNORM divergence at mid-range:
//
//   Orange  (1.0, 0.502, 0.0, 1.0):
//     UNORM bytes: (255, 128,   0, 255)
//     sRGB  bytes: (255, 188,   0, 255)  ← G channel off by 60!
//
//   Rebecca Purple (0.4, 0.2, 0.6, 1.0):
//     UNORM bytes: (102,  51, 153, 255)
//     sRGB  bytes: (171, 113, 201, 255)  ← R off by 69, G off by 62, B off by 48!
//
//   Mid grey (0.5, 0.5, 0.5, 1.0):
//     UNORM bytes: (128, 128, 128, 255)
//     sRGB  bytes: (188, 188, 188, 255)  ← off by 60!
//
//   Pure white (1.0, 1.0, 1.0, 1.0):
//     Both:        (255, 255, 255, 255)  ← identical (control)

struct TestColor
{
	float		r, g, b, a;
	const char*	name;
};

static const TestColor kTestColors[] =
{
	{ 1.0f,  0.502f, 0.0f,  1.0f, "orange" },
	{ 0.4f,  0.2f,   0.6f,  1.0f, "rebecca_purple" },
	{ 0.5f,  0.5f,   0.5f,  1.0f, "mid_grey" },
	{ 1.0f,  1.0f,   1.0f,  1.0f, "white" },
};

static const uint32_t kNumTestColors = 4u;

static uint8_t floatToUnormByte(float v)
{
	return static_cast<uint8_t>(std::floor(v * 255.0f + 0.5f));
}

void checkSupport(Context& context, TestParams params)
{
	context.requireDeviceFunctionality("VK_KHR_maintenance2");

	const VkFormatProperties srgbProps = getPhysicalDeviceFormatProperties(
		context.getInstanceInterface(), context.getPhysicalDevice(), params.srgbFormat);

	if (!(srgbProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
		TCU_THROW(NotSupportedError, "sRGB format does not support transfer src");

	const VkFormatProperties unormProps = getPhysicalDeviceFormatProperties(
		context.getInstanceInterface(), context.getPhysicalDevice(), params.unormFormat);

	if (!(unormProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
		TCU_THROW(NotSupportedError, "UNORM format does not support storage images");
}

void initPrograms(vk::SourceCollections& programCollection, TestParams params)
{
	(void)params;

	// Compute shader writes known colors via imageStore through a UNORM view.
	// layout(rgba8) = SPIR-V Rgba8 = VK_FORMAT_R8G8B8A8_UNORM.
	// No sRGB conversion must happen.
	std::ostringstream src;
	src << "#version 450\n"
		<< "\n"
		<< "layout (local_size_x = 1, local_size_y = 1) in;\n"
		<< "\n"
		<< "layout(binding = 0, rgba8) writeonly uniform image2D u_image;\n"
		<< "\n"
		<< "// Test colors: orange, rebecca purple, mid grey, white\n"
		<< "const vec4 colors[4] = vec4[4](\n"
		<< "    vec4(1.0, 0.502, 0.0, 1.0),\n"    // orange
		<< "    vec4(0.4, 0.2,   0.6, 1.0),\n"    // rebecca purple
		<< "    vec4(0.5, 0.5,   0.5, 1.0),\n"    // mid grey
		<< "    vec4(1.0, 1.0,   1.0, 1.0)\n"     // white (control)
		<< ");\n"
		<< "\n"
		<< "void main(void)\n"
		<< "{\n"
		<< "    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n"
		<< "    int idx = pos.x % 4;\n"
		<< "    imageStore(u_image, pos, colors[idx]);\n"
		<< "}\n";

	programCollection.glslSources.add("comp") << glu::ComputeSource(src.str());
}

tcu::TestStatus testSrgbStorageNoConversion(Context& context, TestParams params)
{
	const DeviceInterface&	vk				= context.getDeviceInterface();
	const VkDevice			device			= context.getDevice();
	Allocator&				allocator		= context.getDefaultAllocator();
	const uint32_t			queueFamilyIndex = context.getUniversalQueueFamilyIndex();
	const VkQueue			queue			= context.getUniversalQueue();

	const uint32_t			width			= kNumTestColors;
	const uint32_t			height			= 1u;
	const VkDeviceSize		pixelSize		= tcu::getPixelSize(mapVkFormat(params.unormFormat));
	const VkDeviceSize		bufferSize		= width * height * pixelSize;

	// --- Create image as sRGB with MUTABLE_FORMAT_BIT + EXTENDED_USAGE_BIT ---
	const VkImageCreateInfo imageCreateInfo =
	{
		VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		nullptr,
		VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
		VK_IMAGE_TYPE_2D,
		params.srgbFormat,
		makeExtent3D(width, height, 1u),
		1u, 1u,
		VK_SAMPLE_COUNT_1_BIT,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		VK_SHARING_MODE_EXCLUSIVE,
		0u, nullptr,
		VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const auto image		= createImage(vk, device, &imageCreateInfo);
	const auto imageAlloc	= allocator.allocate(
		getImageMemoryRequirements(vk, device, *image), MemoryRequirement::Any);
	VK_CHECK(vk.bindImageMemory(device, *image, imageAlloc->getMemory(), imageAlloc->getOffset()));

	// --- Create UNORM view for storage ---
	const VkImageViewUsageCreateInfo viewUsageInfo =
	{
		VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
		nullptr,
		VK_IMAGE_USAGE_STORAGE_BIT,
	};

	const VkImageViewCreateInfo viewCreateInfo =
	{
		VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		&viewUsageInfo,
		0u,
		*image,
		VK_IMAGE_VIEW_TYPE_2D,
		params.unormFormat,
		makeComponentMappingRGBA(),
		makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u),
	};

	const auto imageView = createImageView(vk, device, &viewCreateInfo);

	// --- Readback buffer ---
	const auto buffer		= makeBuffer(vk, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	const auto bufferAlloc	= allocator.allocate(
		getBufferMemoryRequirements(vk, device, *buffer), MemoryRequirement::HostVisible);
	VK_CHECK(vk.bindBufferMemory(device, *buffer, bufferAlloc->getMemory(), bufferAlloc->getOffset()));

	// --- Descriptor set ---
	const auto descriptorPool = DescriptorPoolBuilder()
		.addType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
		.build(vk, device, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, 1u);

	const auto descriptorSetLayout = DescriptorSetLayoutBuilder()
		.addSingleBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
		.build(vk, device);

	const auto descriptorSet = makeDescriptorSet(vk, device, *descriptorPool, *descriptorSetLayout);

	const VkDescriptorImageInfo imageDescriptorInfo = makeDescriptorImageInfo(
		VK_NULL_HANDLE, *imageView, VK_IMAGE_LAYOUT_GENERAL);

	DescriptorSetUpdateBuilder()
		.writeSingle(*descriptorSet, DescriptorSetUpdateBuilder::Location::binding(0u),
					 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageDescriptorInfo)
		.update(vk, device);

	// --- Pipeline ---
	const auto pipelineLayout	= makePipelineLayout(vk, device, *descriptorSetLayout);
	const auto shaderModule		= createShaderModule(vk, device,
		context.getBinaryCollection().get("comp"), 0);
	const auto pipeline = makeComputePipeline(vk, device, *pipelineLayout, *shaderModule);

	// --- Record commands ---
	const auto cmdPool		= createCommandPool(vk, device,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex);
	const auto cmdBuffer	= allocateCommandBuffer(vk, device, *cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	beginCommandBuffer(vk, *cmdBuffer);

	// Transition image to GENERAL for storage
	const VkImageMemoryBarrier imageBarrierToGeneral =
	{
		VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		nullptr,
		0u,
		VK_ACCESS_SHADER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		*image,
		makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u),
	};

	vk.cmdPipelineBarrier(*cmdBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0u, 0u, nullptr, 0u, nullptr, 1u, &imageBarrierToGeneral);

	// Dispatch
	vk.cmdBindPipeline(*cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
	vk.cmdBindDescriptorSets(*cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
		*pipelineLayout, 0u, 1u, &descriptorSet.get(), 0u, nullptr);
	vk.cmdDispatch(*cmdBuffer, width, height, 1u);

	// Barrier: compute write -> transfer read
	const VkImageMemoryBarrier imageBarrierToTransfer =
	{
		VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		nullptr,
		VK_ACCESS_SHADER_WRITE_BIT,
		VK_ACCESS_TRANSFER_READ_BIT,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		*image,
		makeImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u),
	};

	vk.cmdPipelineBarrier(*cmdBuffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		0u, 0u, nullptr, 0u, nullptr, 1u, &imageBarrierToTransfer);

	// Copy image to buffer
	const VkBufferImageCopy copyRegion =
	{
		0u, 0u, 0u,
		makeImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u),
		makeOffset3D(0, 0, 0),
		makeExtent3D(width, height, 1u),
	};

	vk.cmdCopyImageToBuffer(*cmdBuffer, *image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		*buffer, 1u, &copyRegion);

	// Barrier: transfer -> host read
	const VkBufferMemoryBarrier bufferBarrier =
	{
		VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		nullptr,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_HOST_READ_BIT,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		*buffer,
		0u, bufferSize,
	};

	vk.cmdPipelineBarrier(*cmdBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
		0u, 0u, nullptr, 1u, &bufferBarrier, 0u, nullptr);

	endCommandBuffer(vk, *cmdBuffer);
	submitCommandsAndWait(vk, device, queue, *cmdBuffer);

	// --- Verify results ---
	invalidateAlloc(vk, device, *bufferAlloc);

	const uint8_t*	resultBytes	= static_cast<const uint8_t*>(bufferAlloc->getHostPtr());
	tcu::TestLog&	log			= context.getTestContext().getLog();
	bool			passed		= true;

	for (uint32_t x = 0; x < width; ++x)
	{
		const TestColor& tc = kTestColors[x % kNumTestColors];
		const uint8_t expectedR = floatToUnormByte(tc.r);
		const uint8_t expectedG = floatToUnormByte(tc.g);
		const uint8_t expectedB = floatToUnormByte(tc.b);
		const uint8_t expectedA = floatToUnormByte(tc.a);

		const uint8_t actualR = resultBytes[x * 4 + 0];
		const uint8_t actualG = resultBytes[x * 4 + 1];
		const uint8_t actualB = resultBytes[x * 4 + 2];
		const uint8_t actualA = resultBytes[x * 4 + 3];

		// Tolerance of 1 for UNORM rounding.
		// sRGB leakage would produce diffs of 30-70 for mid-range values.
		const int tolerance = 1;

		const int diffR = std::abs((int)actualR - (int)expectedR);
		const int diffG = std::abs((int)actualG - (int)expectedG);
		const int diffB = std::abs((int)actualB - (int)expectedB);
		const int diffA = std::abs((int)actualA - (int)expectedA);

		if (diffR > tolerance || diffG > tolerance || diffB > tolerance || diffA > tolerance)
		{
			log << tcu::TestLog::Message
				<< "FAIL pixel " << x << " (" << tc.name << "):"
				<< " expected RGBA=(" << (int)expectedR << ", " << (int)expectedG
				<< ", " << (int)expectedB << ", " << (int)expectedA << ")"
				<< " got RGBA=(" << (int)actualR << ", " << (int)actualG
				<< ", " << (int)actualB << ", " << (int)actualA << ")"
				<< " diff=(" << diffR << ", " << diffG << ", " << diffB << ", " << diffA << ")"
				<< tcu::TestLog::EndMessage;
			passed = false;
		}
		else
		{
			log << tcu::TestLog::Message
				<< "OK   pixel " << x << " (" << tc.name << "):"
				<< " RGBA=(" << (int)actualR << ", " << (int)actualG
				<< ", " << (int)actualB << ", " << (int)actualA << ")"
				<< tcu::TestLog::EndMessage;
		}
	}

	return passed ? tcu::TestStatus::pass("Passed")
				  : tcu::TestStatus::fail("sRGB conversion applied on UNORM storage view");
}

} // anonymous namespace

tcu::TestCaseGroup* createImageSrgbStorageTests(tcu::TestContext& testCtx)
{
	de::MovePtr<tcu::TestCaseGroup> testGroup(
		new tcu::TestCaseGroup(testCtx, "srgb_storage_no_conversion"));

	const struct
	{
		VkFormat	srgbFormat;
		VkFormat	unormFormat;
		const char*	name;
	} cases[] =
	{
		{ VK_FORMAT_R8G8B8A8_SRGB,			VK_FORMAT_R8G8B8A8_UNORM,			"r8g8b8a8" },
		{ VK_FORMAT_B8G8R8A8_SRGB,			VK_FORMAT_B8G8R8A8_UNORM,			"b8g8r8a8" },
		{ VK_FORMAT_A8B8G8R8_SRGB_PACK32,	VK_FORMAT_A8B8G8R8_UNORM_PACK32,	"a8b8g8r8_pack32" },
	};

	for (const auto& c : cases)
	{
		const TestParams params = { c.srgbFormat, c.unormFormat };
		addFunctionCaseWithPrograms(testGroup.get(), c.name, checkSupport, initPrograms,
									testSrgbStorageNoConversion, params);
	}

	return testGroup.release();
}

} // namespace image
} // namespace vkt
