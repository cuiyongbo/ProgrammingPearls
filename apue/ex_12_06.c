#include "apue.h"

unsigned int sleep_s(unsigned int sec)
{
	time_t start, end;
	struct timeval tv;
	tv.tv_sec = sec;
	tv.tv_usec = 0;
	
	time(&start);
	int n = select(0, NULL, NULL, NULL, &tv);
	if(n == 0) return 0;
	time(&end);
	
	int slept = end - start;
	return (slept >= sec) ? 0 : (sec - slept); 
}
