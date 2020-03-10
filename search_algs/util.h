#pragma once

#define element_count_of_array(a) (sizeof(a)/sizeof(a[0]))

int* genRandomArray(int size);
void freeRandomArray(int* arr);

// end is excluive;
int* duplicateArray(int* src, int begin, int end);
void freeDuplicateArray(int* arr);

bool isSorted(int* arr, int size, bool ascendingSorted);

