#include <iostream>
#include <algorithm>

int main()
{
	int a[] = {1,2,4,1,5,7,3,9};
	int length = sizeof(a)/sizeof(a[0]);
	
	std::nth_element(a, a+length/2, a+length);
	for(int i=0; i<length; i++)
		std::cout << a[i] << " ";
	std::cout << "\n";
	return 0;
}
