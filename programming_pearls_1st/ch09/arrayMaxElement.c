#include <stdio.h>

#define element_of(a) (sizeof(a)/sizeof(a[0]))

int alg1(int* arr, int n)
{
	int max = arr[0];
	for(int i=1; i<n; i++)
	{
		if(max < arr[i])
			max = arr[i];
	}
	return max;
}

/* the size of arr must be n+1*/
int alg2(int* arr, int n)
{
	int i = 0;
	int max;
	while(i < n)
	{
		max = arr[i];
		arr[n] = max;
		i++;
		while(arr[i] < max)
			i++;
	}

	return max;
}

int main()
{
	int a[10] = {1,2,3,4,5,6,7,8,9,};
	
	alg2(a, element_of(a)-1);

	return 0;
}


