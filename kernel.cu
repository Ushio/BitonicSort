#include "typedbuffer.hpp"

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

constexpr int N_ELEM_PER_BLOCK = 64;

#if 1
extern "C" __global__ void kernelMain( TypedBuffer<int> xs_input )
{
	__shared__ int xs[N_ELEM_PER_BLOCK];

	xs[threadIdx.x] = xs_input[blockIdx.x * N_ELEM_PER_BLOCK + threadIdx.x];

	__syncthreads();

	for (uint32_t subSize = 2; subSize <= N_ELEM_PER_BLOCK; subSize *= 2)
	{
		for (uint32_t pairDist = subSize / 2; 0 < pairDist; pairDist /= 2)
		{
			uint32_t i = threadIdx.x;
			uint32_t pair = i ^ pairDist;
			int a = xs[i];
			int b = xs[pair];

			int flip = i > pair;
			int decending = (i / subSize) & 0x1;
			int swap = (b < a) ^ decending ^ flip;
			
			__syncthreads();
			
			int out = swap ? b : a;
			xs[i] = out;

			__syncthreads();
		}
	}

	xs_input[blockIdx.x * N_ELEM_PER_BLOCK + threadIdx.x] = xs[threadIdx.x];
}

#else

extern "C" __global__ void  __launch_bounds__( 32 ) kernelMain( TypedBuffer<int> xs_input )
{
	int x = xs_input[blockIdx.x * N_ELEM_PER_BLOCK + threadIdx.x];

	for (uint32_t subSize = 2; subSize <= N_ELEM_PER_BLOCK; subSize *= 2)
	{
		for (uint32_t pairDist = subSize / 2; 0 < pairDist; pairDist /= 2)
		{
			uint32_t i = threadIdx.x;
			uint32_t pair = i ^ pairDist;

			int a = x;
			int b = __shfl_xor( x, pairDist );

			int opposite = i > pair;
			int decending = (i / subSize) & 0x1;
			int swap = (b < a) ^ decending ^ opposite;

			if( swap )
			{
				x = b;
			}
		}
	}

	xs_input[blockIdx.x * N_ELEM_PER_BLOCK + threadIdx.x] = x;
}

#endif