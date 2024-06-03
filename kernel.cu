#include "typedbuffer.hpp"

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

constexpr int N_ELEM_PER_BLOCK = 64;

__device__
int sqr(int x)
{
	return x * x;
}

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

			__syncthreads();

			if (i < pair)
			{
				int decending = (i / subSize) & 0x1;
				int swap = (b < a) ^ decending;

				if (swap)
				{
					xs[i] = b;
					xs[pair] = a;
				}
			}

			__syncthreads();
		}
	}


	xs_input[blockIdx.x * N_ELEM_PER_BLOCK + threadIdx.x] = xs[threadIdx.x];
}