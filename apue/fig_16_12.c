#include "apue.h"
#include <sys/socket.h>

int initServer(int type, const struct sockaddr* addr, socklen_t addrLen, int queueLen)
{
	int err = 0;
	int fd = socket(addr->sa_family, type, 0);
	if(fd < 0)
		goto errout;
	if(bind(fd, addr, addrLen) < 0)
		goto errout;
	if(type == SOCK_STREAM || type == SOCK_SEQPACKET)
	{
		if(listen(fd, queueLen) < 0 )
			goto errout;
	}
	return fd;

errout:
	err = errno;
	close(fd);
	errno = err;
	err_msg("initServer");
	return -1;
}

