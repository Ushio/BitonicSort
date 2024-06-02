#include <iostream>
#include <vector>
#include <algorithm>

#include <intrin.h>
#define ASSERT_TRUE(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }

std::vector<int> sort_bitonic( std::vector<int> xs0 )
{
	ASSERT_TRUE( __popcnt( xs0.size() ) == 1 );

	int N = xs0.size();
	for (uint32_t subSize = 2; subSize <= N; subSize *= 2)
	{
		for (uint32_t pairDist = subSize / 2; 0 < pairDist; pairDist /= 2)
		{
			for (uint32_t i = 0; i < N; i++)
			{
				uint32_t pair = i ^ pairDist;
				if (i < pair)
				{
					int a = xs0[i];
					int b = xs0[pair];
					int decending = (i / subSize) & 0x1;
					int swap = (b < a) ^ decending;

					if (swap)
					{
						std::swap(a, b);
						xs0[i] = a;
						xs0[pair] = b;
					}
				}
			}
		}
	}

	return xs0;
}

int main()
{
	// Random TEST
	for (int i = 0; i < 100000; i++)
	{
		std::vector<int> xs;
		xs.resize( 2u << ( rand() % 10 ) );

		for (int j = 0; j < xs.size(); j++)
		{
			xs[j] = rand();
		}
		std::vector<int> sorted_xs = sort_bitonic(xs);
		std::sort(xs.begin(), xs.end());

		ASSERT_TRUE(xs == sorted_xs);
	}

	printf("passed :)\n");

	// A simple example
	std::vector<int> xs0 = { 3, 7, 4, 8, 6, 2, 1, 5 };

	int N = xs0.size();
	for( uint32_t subSize = 2 ; subSize <= N ; subSize *= 2 )
	{
		for( uint32_t pairDist = subSize / 2 ; 0 < pairDist; pairDist /= 2 )
		{
			printf("subSize=%d, pairDist=%d\n", subSize, pairDist);

			for( uint32_t i = 0; i < N; i++ )
			{
				uint32_t pair = i ^ pairDist;
				if( i < pair )
				{
					int a = xs0[i];
					int b = xs0[pair];
					int decending = (i / subSize) & 0x1;
					int swap = (b < a) ^ decending;

					if( swap )
					{
						std::swap( a, b );
						xs0[i] = a;
						xs0[pair] = b;
					}

					printf("o ");
				}
				else
					printf("x ");
			}
			printf("\n");
			for (uint32_t i = 0; i < N; i++)
			{
				printf("%d ", xs0[i]);
			}
			printf("\n");
		}
	}


	
	return 0;
}
