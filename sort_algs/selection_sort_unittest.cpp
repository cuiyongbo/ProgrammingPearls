#include "util.h"
#include "sort_algs.h"
#include "gtest/gtest.h"

TEST(SelectionSortTests, basic)
{	
	int size = 100; 
	int* a = genRandomArray(size);
	selectionSort(a, size);
	bool isAscendingSorted = isSorted(a, size, true);
	freeRandomArray(a);
	EXPECT_TRUE(isAscendingSorted);
}

