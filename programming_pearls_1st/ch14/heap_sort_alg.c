#include <stdio.h>

#define swap(a, b) {int t=a;a=b;b=t;}

void siftDown(int* v, int end)
{
	// extend heap(1, n) to heap(0, n)
	int i, c;
	for(i=0;(c=2*i+1)<=end; i=c)
	{
		if(c+1 <= end && v[c+1] > v[c])
			c++;
		if(v[i] >= v[c]) break;
		swap(v[i], v[c]);
	}
}

void siftUp(int* v, int end)
{
	// extend heap(0, n-1) to heap(0, n)
	int i, p;
	for(int i=end; i>0 && v[p=(i-1)/2] < v[i]; i=p)
		swap(v[p], v[i]);
}

void heapsort(int* v, int length)
{
	for(int i=1; i<length; i++)
		siftUp(v, i);

	for(int i=length-1; i>0; i--)
	{
		swap(v[0], v[i]);
		siftDown(v, i-1);
	}
}

int main()
{
	const int n = 10;
	int v[n];
	for(int i=0; i<n; i++)
		v[i] = 10-i;

	heapsort(v, n);

	for(int i=0; i<n; i++)
		printf("%d ", v[i]);
	printf("\n");

	return 0;
}

