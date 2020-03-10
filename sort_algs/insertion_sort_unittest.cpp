#include "util.h"
#include "sort_algs.h"
#include "gtest/gtest.h"

TEST(insertionSortTests, basic)
{	
	int size = 100; 
	int* a = genRandomArray(size);
	insertionSort(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}

TEST(insertionSortTests, noSwap)
{
	int size = 100; 
	int* a = genRandomArray(size);
	insertionSort_noSwap(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}
