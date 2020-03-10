#include "apue.h"
#include <sys/socket.h>
#include <sys/un.h>

#define STALE 30 /* client's name can't be older this (sec), but what for? */

int serv_accept(int listenfd, uid_t* uidptr)
{
	struct sockaddr_un un;
	char* name = (char*)malloc(sizeof(un.sun_path) + 1);
	if(name == NULL)
		return -1;

	socklen_t len = sizeof(un);
	int clifd = accept(listenfd, (struct sockaddr*)&un, &len);
	if(clifd < 0)
	{
		free(name);
		return -2;
	}
	
	len = len - offsetof(struct sockaddr_un, sun_path);
	memcpy(name, un.sun_path, len);
	name[len] = 0;

	int err, rval;
	time_t staletime;
	struct stat statbuf;
	if(stat(name, &statbuf) < 0)
	{
		rval = -3;
		goto errout;
	}	

#ifdef S_ISSOCK
	if(S_ISSOCK(statbuf.st_mode) == 0)
	{
		rval = -4;
		goto errout;
	}
#endif

	if((statbuf.st_mode & (S_IRWXG | S_IRWXO)) ||
		(statbuf.st_mode & S_IRWXU) != S_IRWXU)
	{
		rval = -5;
		goto errout;
	}

	staletime = time(NULL) - STALE;
	if( statbuf.st_atime < staletime ||
		statbuf.st_ctime < staletime ||
		statbuf.st_mtime < staletime )
	{
		rval = -6; /* i-node is too old */
		goto errout;
	}

	if(uidptr != NULL)
		*uidptr = statbuf.st_uid;

	unlink(name);
	free(name);
	return clifd;

errout:
	err = errno;
	close(clifd);
	free(name);
	errno = err;
	return rval;
}

