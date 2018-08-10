#include "gtest/gtest.h"

int binarySearch(int* arr, int size, int val)
{
	int start = 0;
	int end = size - 1;
	if (start > end || arr[start] > val || arr[end] < val)
		return -1;

	while (start <= end)
	{
		int mid = start + (end - start) / 2;
		if (arr[mid] == val)
			return mid;
		else if (arr[mid] > val)
			end = mid - 1;
		else
			start = mid + 1;
	}
	return -1;
}

int lowerBound(int* arr, int size, int val)
{
	int l = 0;
	int r = size;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (arr[mid] < val)
			l = mid + 1;
		else
			r = mid;
	}
	return l;
}

int upperBound(int* arr, int size, int val)
{
	int l = 0;
	int r = size;
	while (l < r)
	{
		int mid = l + (r - l) / 2;
		if (arr[mid] <= val)
			l = mid + 1;
		else
			r = mid;
	}
	return l-1;
}

TEST(BinarySearchTest, binarySearch)
{
	int arr[] = {1,2,3,4,5,6,7,8,9,10};
	int n = element_count_of_array(arr);
	for (int i = 0; i < n; i++)
		EXPECT_EQ(i, binarySearch(arr, n, arr[i]));

	EXPECT_EQ(-1, binarySearch(arr, n, -1));

	int arr2[] = { 0 };
	EXPECT_EQ(-1, binarySearch(arr2, element_count_of_array(arr2), -1));
	EXPECT_EQ(0, binarySearch(arr2, element_count_of_array(arr2), 0));
	EXPECT_EQ(-1, binarySearch(arr2, element_count_of_array(arr2), 1));
}

TEST(BinarySearchTest, lowerBound)
{
	int arr[] = { 1, 2, 4, 4, 5, 5, 5, 8, 9, 10 };
	int n = element_count_of_array(arr);
	EXPECT_EQ(2, lowerBound(arr, n, 4));
	EXPECT_EQ(4, lowerBound(arr, n, 5));
	EXPECT_EQ(n, lowerBound(arr, n, 100));
	EXPECT_EQ(0, lowerBound(arr, n, 0));
	EXPECT_EQ(0, lowerBound(arr, n, 1));
}

TEST(BinarySearchTest, upperBound)
{
	int arr[] = { 1, 2, 4, 4, 5, 5, 5, 8, 9, 10 };
	int n = element_count_of_array(arr);
	EXPECT_EQ(3, upperBound(arr, n, 4));
	EXPECT_EQ(6, upperBound(arr, n, 5));
	EXPECT_EQ(n-1, upperBound(arr, n, 100));
	EXPECT_EQ(-1, upperBound(arr, n, 0));
	EXPECT_EQ(0, upperBound(arr, n, 1));
}