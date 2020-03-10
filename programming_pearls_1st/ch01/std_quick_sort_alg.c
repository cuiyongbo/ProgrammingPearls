#include <stdio.h>
#include <stdlib.h>

#define NMAX	1024*1024

int intcomp(const void* x, const void* y)
{
	int a = *(int*)x;
	int b = *(int*)y;
	return (a > b) - (a < b);
}

int main()
{
	int n = 0;
	int a[NMAX];
	while(n<NMAX && scanf("%d", a+n) != EOF)
		n++;

	qsort(a, n, sizeof(int), intcomp);

	for(int i=0; i<n; i++)
		printf("%d\n", a[i]);
	return 0;
}

