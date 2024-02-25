#include "apue.h"
#include <sys/socket.h>
#include <sys/types.h>

int initServer(int type, const struct sockaddr* addr, socklen_t addrlen, int queuelen)
{
	int err = 0;
	int reuse = 1;
	int fd = socket(addr->sa_family, type, 0);
	if(fd < 0)
		return -1;
	if(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &resue), sizeof(int) < 0)
		goto errout;
	if(bind(fd, addr, addrlen) < 0)
		goto errout;
	if(type == SOCK_STREAM || type == SOCK_SEQPACKET)
	{
		if(listen(fd, queuelen) < 0)
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
