#include "apue.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#define QUEUE_LENGTH	10

int serv_listen(const char* name)
{
	struct sockaddr_un un;
	if(strlen(name) > sizeof(un.sun_path))
	{
		errno = ENAMETOOLONG;
		return -1;
	}

	int fd = socket(AF_LOCAL, SOCK_STREAM, 0);
	if(fd < 0)
		return -2;

	unlink(name);

	memset(&un, 0, sizeof(un));
	un.sun_family = AF_LOCAL;
	strcpy(un.sun_path, name);
	int len = offsetof(struct sockaddr_un, sun_path) + strlen(name);
	
	int err, rval;
	if(bind(fd, (struct sockaddr*)&un, len) < 0)
	{
		rval = -3;
		goto errout;
	}

	if(listen(fd, QUEUE_LENGTH) < 0)
	{
		rval = -4; 
		goto errout;
	}
	return fd;
	
errout:
	err = errno;
	close(fd);
	errno = err;
	return rval;
}

