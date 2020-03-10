#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define element_of(a)	sizeof(a)/sizeof(a[0])
#define MAXN	10000000
#define Larger_Value(a, b)	((a)>(b)?(a):(b))

int n;
float x[MAXN];

float alg1()
{
	float maxsofar = x[0];
	for(int i=0; i<n; ++i)
	{
		for(int j=i; j<n; ++j) 
		{
			float sum = 0;
			for (int k=i; k<=j; ++k)
				sum += x[k]; // sum of x[i,j]
			maxsofar = Larger_Value(sum, maxsofar);
		}
	}
	return maxsofar;
}

float alg2()
{
	float maxsofar = x[0];
	for(int i=0; i<n; ++i)
	{
		float sum = 0;
		for(int j=i; j<n; ++j) 
		{
			sum += x[j]; // sum of x[i,j]
			maxsofar = Larger_Value(sum, maxsofar);
		}
	}
	return maxsofar;
}

float cumvec[MAXN+1];
float alg2b()
{
	float* cumarr = cumvec+1;
	cumarr[-1] = 0;
	for(int i=0; i<n; ++i)
		cumarr[i] = cumarr[i-1] + x[i];

	float maxsofar = x[0];
	for(int i=0; i<n; ++i)
	{
		float sum = 0;
		for(int j=i; j<n; ++j) 
		{
			sum = cumarr[j]-cumarr[i-1]; // sum of x[i,j]
			maxsofar = Larger_Value(sum, maxsofar);
		}
	}
	return maxsofar;
}

float recmax(int l, int u)
{
	if(l>u) // zero element
		return 0;

	if(l==u) // one element
		return Larger_Value(0, x[l]);


	int m = l + (u-l)/2;
	float lmax, rmax, sum;

	// calculate sum crossing middle
	// find max crossing to left
	lmax = sum = 0;	
	for(int i=m; i >= l; --i)
	{
		sum += x[i];
		lmax = Larger_Value(lmax, sum);
	}

	// find max crossing to right
	rmax = sum = 0;
	for(int i=m+1; i<=u; ++i)
	{
		sum += x[i];
		rmax = Larger_Value(rmax, sum);
	}

	return Larger_Value(lmax+rmax, Larger_Value(recmax(l, m), recmax(m+1, u)));
}

float alg3()
{
	return recmax(0, n-1);
}

float alg4()
{
	float maxsofar = 0;
	float maxendinghere = 0;
	for(int i=0; i<n; ++i)
	{
		maxendinghere += x[i];
		maxendinghere = Larger_Value(maxendinghere, 0);
		maxsofar = Larger_Value(maxsofar, maxendinghere);
	}
	return maxsofar;	
}

// Fill x[n] with reals uniform on [-1, 1]
void sprinkle()
{
	for (int i=0; i<n; ++i)
		x[i] = 1-2*((float)rand()/RAND_MAX);
}

int main()
{
	int algnum;
	int start, clicks;
	float thisans;
	
	while(scanf("%d %d", &algnum, &n) != EOF)
	{
		sprinkle();
		start = clock();
		thisans = -1;
		switch(algnum)
		{
			case 1: thisans = alg1(); break;
			case 2: thisans = alg2(); break;
			case 22: thisans = alg2b(); break;
			case 3: thisans = alg3(); break;
			case 4: thisans = alg4(); break;
			default: break;

		}
		clicks = clock() - start;
		printf("%d\t%d\t%f\t%d\t%f\n", algnum, n, thisans, 
				clicks, clicks/(float)CLOCKS_PER_SEC);
		if(alg4() != thisans)
			printf("maxsum error: mismatch with alg4: %f\n", alg4());
		if(alg1() != alg2b())
			printf("maxsum error: mismatch with alg2b: %f\n", alg2b());
	}

	return 0;
}

