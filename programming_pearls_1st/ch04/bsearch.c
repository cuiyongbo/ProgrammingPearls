#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define element_count(array) sizeof(array)/sizeof(array[0])

typedef int DataType;

size_t bsearch_local(DataType key, DataType* arr, size_t count)
{
	size_t start = 0;
	size_t end = count -1;
	size_t pos = -1;

	while(start <= end) {
		size_t mid = start + (end-start)/2;
		// make sure the array is sorted
		assert(arr[start] <= arr[mid] && arr[mid] <= arr[end]);

		if(arr[mid] < key) {
			start = mid + 1;
		} else if(arr[mid] == key) {
			pos = mid;
			break;
		} else {
			end = mid - 1;
		}
	}

	return pos;
}


int main(int argc, char* argv[])
{
	DataType key;
	size_t count; 
	size_t pos = -1;
	while(scanf("%lu %d", &count, &key) != EOF) {
		DataType* arr = (DataType*)malloc(count * sizeof(DataType));
		for(int i = 0; i<count; ++i)
			arr[i] = i*10;
		pos = bsearch_local(key, arr, count);
		printf("%lu\n", pos);
		free(arr);
	}

	return 0;
}


