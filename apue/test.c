#include "apue.h"
#include <arpa/inet.h>


int main()
{
	uint32_t a = 0x04030201;
	uint8_t* bytes = (uint8_t*)&a;
	printf("host byteorder: %u, %u, %u, %u\n", 
			bytes[0], bytes[1], bytes[2], bytes[3]);
	
	uint32_t b = htonl(a);
	bytes = (uint8_t*)&b;
	printf("network byteorder: %u, %u, %u, %u\n", 
			bytes[0], bytes[1], bytes[2], bytes[3]);
		
	return 0;
}
