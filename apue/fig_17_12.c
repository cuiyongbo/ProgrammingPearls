#include "apue.h"

int send_err(int fd, int errcode, const char* msg)
{
	int n = (int)strlen(msg);
	if(n > 0)
	{
		if(write(fd, msg, n) != n)
		 return -1;
	}

	if(errcode >= 0)
		errcode = -1;

	if(send_fd(fd, errcode) < 0)
		return -1;

	return 0;
}
