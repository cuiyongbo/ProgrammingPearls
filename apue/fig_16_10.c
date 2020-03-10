#include "apue.h"
#include <sys/socket.h>

#define MAXSLEEP 128

int connect_with_retry(int sock, const struct sockaddr* addr, socklen_t address_len)
{
	for(int numsec = 1; numsec <= MAXSLEEP; numsec <<= 1)
	{
		if(connect(sock, addr, address_len) == 0)
		{
			/*Connection accepted*/
			return 0;
		}
		/*Delay before retrying*/
		if(numsec <= MAXSLEEP/2)
			sleep(numsec);
	}
	return -1;
}

