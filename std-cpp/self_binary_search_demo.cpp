#include <iostream>
#include <algorithm>
#include <iterator>

int lowerBound(int* a, int n, int k)
{
	int first = 0;
	int count = n;
	while(count > 0)
	{
		int count2 = count/2;
		int m = first + count2;
		if(a[m] < k)
		{
			first = ++m;
			count -= (count2+1);
		}
		else
		{
			count = count2;
		}
	}
	return first;
}

int upperBound(int* a, int n, int k)
{
	int first = 0;
	int count = n;
	while(count > 0)
	{
		int count2 = count/2;
		int m = first + count2;
		if(a[m] <= k)
		{
			first = ++m;
			count -= (count2+1);
		}
		else
		{
			count = count2;
		}
	}
	return first;
}

int lowerBound_1(int* a, int n, int k)
{
	int first = 0;
	int last = n;
	while(first < last)
	{
		int m = first + (last-first)/2;
		if(a[m] < k)
		{
			first = ++m;
		}
		else
		{
			last = m;
		}
	}
	return first;
}

int upperBound_1(int* a, int n, int k)
{
	int first = 0;
	int last = n;
	while(first < last)
	{
		int m = first + (last-first)/2;
		if(a[m] <= k)
		{
			first = ++m;
		}
		else
		{
			last = m;
		}
	}
	return first;
}

bool binarySearch(int* a, int n, int k)
{
	int m = lowerBound_1(a, n, k);
	return m<n && a[m]==k;
}

int main()
{
	int a[] = {1,2,3,4,5,5,5,8,9,10};
	int count = sizeof(a)/sizeof(a[0]);
	std::copy(a, a+count, std::ostream_iterator<int>(std::cout, " "));
	std::cout << "\n";
	
	std::cout << "Lower bound: " << lowerBound(a, count, 5) << '\n';
	std::cout << "Upper bound: " << upperBound(a, count, 5) << '\n';
	std::cout << "Lower bound: " << lowerBound_1(a, count, 5) << '\n';
	std::cout << "Upper bound: " << upperBound_1(a, count, 5) << '\n';
	std::cout << "Binary search: " << std::boolalpha << binarySearch(a, count, 5) << "\n";
}
