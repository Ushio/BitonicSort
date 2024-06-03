#include <iostream>
#include <memory>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include "shader.hpp"
#include "typedbuffer.hpp"

constexpr int N_ELEM_PER_BLOCK = 64;

int main() {
	if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
	{
		printf("failed to init..\n");
		return 0;
	}
	int deviceIdx = 0;

	oroError err;
	err = oroInit(0);
	oroDevice device;
	err = oroDeviceGet(&device, deviceIdx);
	oroCtx ctx;
	err = oroCtxCreate(&ctx, 0, device);
	oroCtxSetCurrent(ctx);

	oroStream stream = 0;
	oroStreamCreate(&stream);
	oroDeviceProp props;
	oroGetDeviceProperties(&props, device);

	bool isNvidia = oroGetCurAPI(0) & ORO_API_CUDADRIVER;

	printf("Device: %s\n", props.name);
	printf("Cuda: %s\n", isNvidia ? "Yes" : "No");


	std::string baseDir = "../"; /* repository root */

	std::vector<std::string> options;
	options.push_back("-I" + baseDir);

	if (isNvidia)
	{
		// options.push_back("--gpu-architecture=compute_70");
		options.push_back(NV_ARG_LINE_INFO);
	}
	else
	{
		options.push_back(AMD_ARG_LINE_INFO);
	}
	// Debug
	//if( isNvidia )
	//{
	//	options.push_back("-G");
	//}
	//else
	//{
	//	options.push_back("-O0");
	//}

	Shader shader((baseDir + "\\kernel.cu").c_str(), "kernel.cu", options );
	// int blocks = 1;
	int blocks = 1024 * 1024 * 4;

	std::vector<int> bufferCPU(N_ELEM_PER_BLOCK * blocks);
	for (int i = 0; i < bufferCPU.size(); i++)
	{
		bufferCPU[i] = rand();
	}

	std::vector<int> reference = bufferCPU;
	for (int i = 0; i < blocks; i++)
	{
		auto beg = reference.begin() + i * N_ELEM_PER_BLOCK;
		std::sort(beg, beg + N_ELEM_PER_BLOCK);
	}


	TypedBuffer<int> bufferGPU(TYPED_BUFFER_DEVICE);
	bufferGPU.allocate(N_ELEM_PER_BLOCK * blocks);
	// for (int j = 0; j < 128; j++)
	for(;;)
	{
		oroMemcpyHtoD(bufferGPU.data(), bufferCPU.data(), bufferGPU.bytes());
		
		OroStopwatch sw(stream);
		sw.start();

		shader.launch("kernelMain",
			ShaderArgument().ptr(&bufferGPU),
			blocks, 1, 1, N_ELEM_PER_BLOCK, 1, 1, stream);

		sw.stop();
		oroStreamSynchronize(stream);

		printf("%.3f ms\n", sw.getMs());

		TypedBuffer<int> outputs = bufferGPU.toHost();

		for (int i = 0; i < reference.size(); i++)
		{
			SH_ASSERT(reference[i] == outputs[i]);
		}
	}
	return 0;
}
