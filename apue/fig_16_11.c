#include "apue.h"
#include <sys/socket.h>

#define MAXSLEEP 128

int connect_with_retry(int domain, int type, int protocol, 
				const struct sockaddr* addr, socklen_t address_len)
{
	for(int numsec = 1; numsec <= MAXSLEEP; numsec <<= 1)
	{
		/*Try to connect with exponential backoff*/
		int fd = socket(domain, type, protocol);
		if(fd < 0)
			return -1;
		if(connect(fd, addr, address_len) == 0)
		{
			/*Connection accepted*/
			return fd;
		}
		close(fd);

		/*Delay before retrying*/
		if(numsec <= MAXSLEEP/2)
			sleep(numsec);
	}
	return -1;
}

