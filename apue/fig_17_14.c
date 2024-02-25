#include "apue.h"
#include <sys/socket.h>

#define  CONTROL_MSG_LEN CMSG_LEN(sizeof(int))

static struct cmsghdr* cmptr = NULL;

int recv_fd(int fd, UnixDomainSocketUserFunc func)
{
	char buf[BUFSIZ];
	struct iovec iov[1];
	struct msghdr msg;

	int status = -1;
	for(;;)
	{
		iov[0].iov_base = buf;
		iov[0].iov_len = sizeof(buf);
		msg.msg_iov = iov;
		msg.msg_iovlen = 1;
		msg.msg_name = NULL;
		msg.msg_namelen = 0;
	
		if(cmptr == NULL && (cmptr = malloc(CONTROL_MSG_LEN)) == NULL)
			return -1;
		msg.msg_control = cmptr;
		msg.msg_controllen = CONTROL_MSG_LEN;
		int nr = recvmsg(fd, &msg, 0);
		if(nr < 0)
		{
			err_msg("recvmsg error");
			return -1;
		}
		else if(nr == 0)
		{
			err_msg("connection closed by server");
			return -1;
		}

		int newfd;
		for(char* ptr=buf; ptr < buf+nr;)
		{
			if(*ptr++ == 0)
			{
				if(ptr != buf+nr-1)
					err_dump("message format error");
				status = *ptr & 0xFF;
				if(status == 0)
				{
					if(msg.msg_controllen != CONTROL_MSG_LEN)
						err_dump("status = 0 but no fd");
					newfd = *(int*)CMSG_DATA(cmptr);
				}
				else
				{
					newfd = -status;
				}
				nr -= 2;
			}
		}
		if(nr > 0 && (*func)(STDERR_FILENO, buf, nr) != nr)
			return -1;
		if(status >= 0)
			return newfd;
	}
}

