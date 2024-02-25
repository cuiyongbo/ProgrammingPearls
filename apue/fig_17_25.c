#include "open.h"
#include <sys/uio.h> // struct iovec

// Open the file by sending the name and oflag
// to the connection server and reading a file
// descriptor back.

int csopen(char* name, int oflag)
{
	static int csfd = -1;
	if(csfd < 0)
	{
		csfd=cli_conn(CS_OPEN);
		if(csfd < 0)
		{
			err_msg("cli_conn error");
			return -1;
		}
	}
	char buf[10];
	sprintf(buf, " %d", oflag);
	struct iovec iov[3];
	iov[0].iov_base = CL_OPEN " ";
	iov[0].iov_len = strlen(CL_OPEN) + 1;
	iov[1].iov_base = name;
	iov[1].iov_len = strlen(name);
	iov[2].iov_base = buf;
	iov[2].iov_len = strlen(buf) + 1;
	ssize_t len = iov[0].iov_len + iov[1].iov_len + iov[2].iov_len;
	if(writev(csfd, iov, 3) != len)
	{
		err_msg("writev error");
		return -1;
	}
	
	return recv_fd(csfd, write);
}

