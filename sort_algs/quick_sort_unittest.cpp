#include "util.h"
#include "sort_algs.h"
#include "gtest/gtest.h"

TEST(QuickSortTests, basic)
{	
	int size = 100; 
	int* a = genRandomArray(size);
	quickSort(a, 0, size-1);
	bool isAscendingSorted = isSorted(a, size, true);
	EXPECT_TRUE(isAscendingSorted);
	freeRandomArray(a);
}

TEST(QuickSortTests, hoarePartition)
{	
	int size = 100; 
	int* a = genRandomArray(size);
	quickSort_hoare(a, 0, size-1);
	bool isAscendingSorted = isSorted(a, size, true);
	EXPECT_TRUE(isAscendingSorted);
	freeRandomArray(a);
}

TEST(QuickSortTests, threeWayPartition)
{	
	int size = 100; 
	int* a = genRandomArray(size);
	quickSort_threeWayPartition(a, 0, size-1);
	bool isAscendingSorted = isSorted(a, size, true);
	EXPECT_TRUE(isAscendingSorted);
	freeRandomArray(a);
}
