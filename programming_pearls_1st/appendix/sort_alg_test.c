#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXN 10000000

typedef int DataType;
DataType realx[MAXN];
int* x = realx;
int n;

void swap(int i, int j)
{
	DataType t = x[i];
	x[i] = x[j];
	x[j] = t;
}

int randint(int l, int u) { return l + (RAND_MAX*rand() + rand()) % (u-l+1); }

int intcomp(int* x, int* y) { return *x - *y;}

void selsort()
{
	int i, j;
	for(i=0; i<n-1; i++)
		for(j=i; j<n; j++)
			if(x[j] < x[i])
				swap(i, j);
}

void shellsort()
{
	int i, j, h;
	for(h=1; h<n; h=3*h+1)
		;

	for(;;)
	{
		h /= 3;
		if(h<1) break;
		for(i=h; i<n; i++)
		{	
			for(j=i; j>=h; j-=h)
			{
				if(x[j-h] < x[j]) break;
				swap(j-h, j);
			}
		}
	}
}

int main(void)
{

	return 0;
}
