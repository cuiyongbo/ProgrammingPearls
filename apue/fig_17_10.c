#include "apue.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#define CLI_PATH "/var/tmp/"
#define CLI_PERM S_IRWXU

int cli_conn(const char* name)
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

	int len, err, rval, do_unlink = 0;
	memset(&un, 0, sizeof(un));
	un.sun_family = AF_LOCAL;
	sprintf(un.sun_path, "%s%05ld", CLI_PATH, (long)getpid());
	len = offsetof(struct sockaddr_un, sun_path) + strlen(un.sun_path);

	unlink(un.sun_path);
	if(bind(fd, (struct sockaddr*)&un, len) < 0)
	{
		rval = -3;
		goto errout;
	}

	if(chmod(un.sun_path, CLI_PERM) < 0)
	{
		rval = -4;
		do_unlink = 1;
		goto errout;
	}

	memset(&un, 0, sizeof(un));
	un.sun_family = AF_LOCAL;
	strcpy(un.sun_path, name);
	len = offsetof(struct sockaddr_un, sun_path) + strlen(name);
	if(connect(fd, (struct sockaddr*)&un, len) < 0)
	{
		rval = -5;
		do_unlink = 1;
		goto errout;
	}

	return fd;
	
errout:
	err = errno;
	close(fd);
	if(do_unlink)
		unlink(un.sun_path);
	errno = err;
	return rval;
}

