#include "apue.h"
#include <sys/socket.h>

#define RIGHTLEN CMSG_LEN(sizeof(int))
#define  CONTROL_MSG_LEN (RIGHTLEN)

static struct cmsghdr* cmptr = NULL;

int send_fd(int fd, int fd_to_send)
{
	char buf[2];
	struct iovec iov[1];
	iov[0].iov_base = buf;
	iov[0].iov_len = 2;
	
	struct msghdr msg;
	msg.msg_iov = iov;
	msg.msg_iovlen = 1;
	msg.msg_name = NULL;
	msg.msg_namelen = 0;

	if(fd_to_send < 0)
	{
		msg.msg_control = NULL;
		msg.msg_controllen = 0;
		buf[1] = -fd_to_send;
		if(buf[1] == 0)
			buf[1] = 1;
	}
	else
	{
		if(cmptr == NULL && (cmptr = malloc(CONTROL_MSG_LEN)) == NULL)
			return -1;
		msg.msg_control = cmptr;
		msg.msg_controllen = CONTROL_MSG_LEN;
		cmptr->cmsg_level = SOL_SOCKET;
		cmptr->cmsg_type = SCM_RIGHTS;
		cmptr->cmsg_len = RIGHTLEN;
		*(int*)CMSG_DATA(cmptr) = fd_to_send;
		buf[1] = 0;
	}
	buf[0] = 0;
	if(sendmsg(fd, &msg, 0) != 2)
		return -1;
	return 0;
}

