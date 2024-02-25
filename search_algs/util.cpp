#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int* genRandomArray(int size)
{
    srand((unsigned)time(NULL));
    int* arr = new int[size];
    for(int i=0; i<size; i++)
        arr[i] = rand();    
    return arr;
}

void freeRandomArray(int* arr) { delete arr; }

// end is excluive;
int* duplicateArray(int* src, int begin, int end)
{
    int* duplicate = new int[end-begin];
    memmove(duplicate, src, sizeof(int)*(end-begin));
    return duplicate; 
}

void freeDuplicateArray(int* arr) { delete[] arr;}

bool isSorted(int* arr, int size, bool ascendingSorted)
{
    if(size < 2)
        return true;
    
    for(int i=0; i<size-1; i++)
    {   
        bool sorted = arr[i] <= arr[i+1];
        if(sorted != ascendingSorted)
            return false;
    }   
    return true;
}


