#include "ppearls.h"

#define max(a, b) ((a)>(b)?(a):(b))

/*
	Assume the sum of zero-element subarray is zero
	and the sum of max subarray has to be nonnegative. 
*/

// O(n^3)
int alg1(int* a, int count)
{
	int maxsum = 0;
	for(int i=0; i<count; i++)
	{
		for(int j=i; j<count; j++)
		{
			int maxsofar = 0;
			for(int k=i; k<=j; k++)
				maxsofar += a[k];

			maxsum = max(maxsum, maxsofar);
		}
	}
	return maxsum;
}

int alg2(int* a, int count)
{
	int maxsum = 0;
	for(int i=0; i<count; i++)
	{
		int maxsofar = 0;
		for(int j=i; j<count; j++)
		{
			maxsofar += a[j];
			maxsum = max(maxsum, maxsofar);
		}
	}
	return maxsum;
}

int alg2b(int* a, int count)
{
	int* cumarr = (int*)malloc((count+1) * sizeof(int));
	cumarr[0] = 0;
	for(int i=1; i<count+1; i++)
		cumarr[i] = cumarr[i-1]+a[i-1];
	
	int maxsofar, maxsum = 0;
	for(int i=0; i<count; i++)
	{
		for(int j=i+1; j<count+1; j++)
		{
			maxsofar = cumarr[j] - cumarr[i]; // sum[i, j)
			maxsum = max(maxsum, maxsofar);
		}
	}

	free(cumarr);
	return maxsum;
}

int genRandomArray(int* a, int count)
{
	srand((unsigned)time(NULL));
	for(int i=0; i<count; i++)
		a[i] = rand();
}

int main()
{
	int n;
	printf("Enter array size: ");
	if(scanf("%d", &n) == EOF)
	{
		n = 10;
	}
	int* a = (int*)malloc(n*sizeof(int));
	genRandomArray(a, n);
	
	printf("alg1:  %d\n", alg1(a, n));
	printf("alg2:  %d\n", alg2(a, n));
	printf("alg2b: %d\n", alg2b(a, n));

	free(a);
	return 0;
}

