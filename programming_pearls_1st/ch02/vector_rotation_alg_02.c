#include <stdio.h>

int greatestCommonDivisor(int a, int b)
{
	int gcd = 1;
	int loop = a > b ? b : a;
	for(int i=2; i<=loop; i++)
	{
		if(a%i==0 && b%i == 0)
			gcd = i;
	}
	return gcd;
}

// https://en.wikipedia.org/wiki/Euclidean_algorithm
int greatestCommonDivisor_recursion(int a, int b)
{
	return b == 0 ? a : greatestCommonDivisor(b, a%b);
}

int greatestCommonDivisor_division(int a, int b)
{
	while(a != 0)
	{
		int t = b;
		b = a % b;
		a = t;
	}
	return a;
}

int greatestCommonDivisor_substraction(int a, int b)
{
	while(a != b)
	{
		if(a > b)
			a = a - b;
		else
			b = b - a;
	}
	return a;
}

void rotate(int* x, int length, int pivot)
{
	int gcd = greatestCommonDivisor(pivot, length);
	for(int i=0; i<gcd; i++)
	{
		int t = x[i];
		int k, j=i;
		for(;;)
		{
			k = j + pivot;
			if(k >= length)
				k -= length;
			if(k == i)
				break;
			x[j] = x[k];
			j = k; 	
		}
		x[j] = t;
	}
}


void printArray(int* arr, int length, const char* msg)
{
	printf("%s:\n", msg);
	for(int i=0; i<length; i++) printf("%d ", arr[i]);
	printf("\n");
}

int main()
{
	const int n = 10;
	int a[n];
	for(int i=0; i<n; i++) a[i] = i;

	printArray(a, n, "Before rotate(3)");
	rotate(a, n, 3);
	printArray(a, n, "After rotate(3)");

	return 0;
}

