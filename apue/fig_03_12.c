#include "apue.h"

void set_fl(int fd, int flags)
{
	int val = fcntl(fd, F_GETFL, 0);
	if(val < 0)
		err_sys("fcntl F_GETFL error");
	val |= flags;
	if(fcntl(fd, F_SETFL, val)<0)
		err_sys("fcntl F_SETFL error");
}

