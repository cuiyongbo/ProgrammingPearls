#include "util.h"
#include "sort_algs.h"
#include "gtest/gtest.h"

TEST(heapSortTests, siftDown)
{	
	int size = 1000; 
	int* a = genRandomArray(size);
	heapSort_siftDown(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}

TEST(heapSortTests, siftUp)
{
	int size = 1000; 
	int* a = genRandomArray(size);
	heapSort_siftUp(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}
