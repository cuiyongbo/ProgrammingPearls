#pragma once

int* genRandomArray(int size);
void freeRandomArray(int* arr);

// end is excluive;
int* duplicateArray(int* src, int begin, int end);
void freeDuplicateArray(int* arr);

bool isSorted(int* arr, int size, bool ascendingSorted);

