#include <stdio.h>

typedef int DataType;

size_t alg(DataType* arr, int* maskArr, size_t num)
{
	size_t p1, p2;
	p1 = p2 = 0;
	while(p2 != num)
	{
		// find the first ZERO from p1
		while(p1 != num && maskArr[p1] == 1)
			p1++;
	
		if(p1 == num)
			break;
		
		// now maskArr[p1] = 0
		
		// find the first ONE from p2
		p2 = p1 + 1; // next time p2 should be start from the previous p2
		while(p2 != num && maskArr[p2] == 0)
			p2++;

		if(p2 == num)
			break;

		// replace arr[p1] with arr[p2], and change the mask
		arr[p1] = arr[p2];
		maskArr[p1] = 1;
		maskArr[p2] = 0;

		p1++; 
		p2++;
	}

	return p1; // the number of elements left arr[0:p1)
}



