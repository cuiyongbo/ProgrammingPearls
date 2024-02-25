#include "ppearls.h"

/*
	Assume the sum of zero-element subarray is zero
	and the sum of max subarray has to be nonnegative. 
*/

// O(n^3)
float alg1(float* a, int count)
{
	float maxsum = 0;
	for(int i=0; i<count; i++)
	{
		for(int j=i; j<count; j++)
		{
			float maxsofar = 0;
			for(int k=i; k<=j; k++) maxsofar += a[k];
			maxsum = max(maxsum, maxsofar);
		}
	}
	return maxsum;
}

float alg2(float* a, int count)
{
	float maxsum = 0;
	for(int i=0; i<count; i++)
	{
		float maxsofar = 0;
		for(int j=i; j<count; j++)
		{
			maxsofar += a[j];
			maxsum = max(maxsum, maxsofar);
		}
	}
	return maxsum;
}

// there may be some precison differences between alg2() and alg2b(),
// for alg2b() pre-calculates summation, and then does some substractions to
// get subarray sums.
float alg2b(float* a, int count)
{
	float* cumarrvec = (float*)malloc((count+1) * sizeof(float));
	float* cumarr = cumarrvec + 1; 
	cumarr[-1] = 0;
	for(int i=0; i<count; i++)
		cumarr[i] = cumarr[i-1]+a[i];
	
	float maxsofar, maxsum = 0;
	for(int i=0; i<count; i++)
	{
		for(int j=i; j<count; j++)
		{
			maxsofar = cumarr[j] - cumarr[i-1]; // sum[i, j]
			maxsum = max(maxsum, maxsofar);
		}
	}

	free(cumarrvec);
	return maxsum;
}

// there may be some precision differernces.
float findMaxCrossSum(float* a, int low, int mid, int high)
{
	float maxleft, maxright, sum;
	maxleft = sum = 0;
	for(int i=mid; i>=low; i--)
	{
		sum += a[i];
		maxleft = max(maxleft, sum);
	}
	maxright = sum = 0;
	for(int j=mid+1; j<=high; j++)
	{
		sum += a[j];
		maxright = max(maxright, sum);
	}
	return maxleft+maxright;
}

float alg3(float* a, int low, int high)
{
	if(low == high)
		return a[low];

	int mid = low + (high-low)/2;
	//printf("%f %f %f\n", low, mid, high);
	float maxleft = alg3(a, low, mid);
	float maxright = alg3(a, mid+1, high);	 
	float crossmax = findMaxCrossSum(a, low, mid, high);
	return max(crossmax, max(maxleft, maxright));
}

float alg4(float* a, int count)
{
	float maxsum, maxsofar;
	maxsum = maxsofar = 0;
	for(int i=0; i<count; i++)
	{
		maxsofar = max(maxsofar+a[i], 0);
		maxsum = max(maxsofar, maxsum);
	}
	return maxsum;
}

void genRandomArray_2(float* a, int length)
{
 	srand((unsigned)time(NULL));
	for(int i=0; i<length; i++)
	{
		a[i] = 1 - 2* ((float)rand()/RAND_MAX);
	}
}

int main()
{
	
/*
	int n;
	printf("Enter array size: ");
	if(scanf("%d", &n) == EOF)
	{
		n = 10;
	}
	float* a = (float*)malloc(n*sizeof(float));
	genRandomArray_2(a, n);	
	printf("alg1:  %f\n", alg1(a, n));
	printf("alg2:  %f\n", alg2(a, n));
	printf("alg2b: %f\n", alg2b(a, n));
	printf("alg3:  %f\n", alg3(a, 0, n-1));
	printf("alg4:  %f\n", alg4(a, n));
	assert(alg1(a, n) == alg2(a, n));
	assert(alg1(a, n) == alg4(a, n));
	free(a);
*/
	int loop, n;
	printf("Enter loop count: ");
	if(scanf("%d", &loop) == EOF || n<10)
	{
		loop = 20;
	}
	
	for(int i=10; i<loop; i+=5)
	{
		n  = i;
		float* a = (float*)malloc(n*sizeof(float));
		genRandomArray_2(a, n);	
//		printf("alg1:  %f\n", alg1(a, n));
//		printf("alg2:  %f\n", alg2(a, n));
//		printf("alg2b: %f\n", alg2b(a, n));
//		printf("alg3:  %f\n", alg3(a, 0, n-1));
//		printf("alg4:  %f\n", alg4(a, n));
		float ans1 = alg1(a, n);
		float ans2 = alg2(a, n);
		float ans2b = alg2b(a, n);
		float ans3 = alg3(a, 0, n-1);
		float ans4 = alg4(a, n);
		assert(ans1 == ans2);
		assert(ans1 == ans4);
		assert(((int)ans3*1e5) == ((int)ans4*1e5));
		assert(((int)ans2b*1e5) == ((int)ans2*1e5));
//		assert(alg1(a, n) == alg2b(a, n));
		free(a);
	}
	return 0;
}

