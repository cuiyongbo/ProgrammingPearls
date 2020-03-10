#include "util.h"
#include "sort_algs.h"
#include "gtest/gtest.h"

TEST(mergeSortTests, bottomUp)
{	
	int size = 100; 
	int* a = genRandomArray(size);
	mergeSort_bottomUp(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}

TEST(mergeSortTests, topDown)
{
	int size = 100; 
	int* a = genRandomArray(size);
	mergeSort_topDown(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}
