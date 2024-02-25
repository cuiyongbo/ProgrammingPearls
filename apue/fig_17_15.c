#include "apue.h"
#include <sys/socket.h>

#if defined(SCM_CREDS)
#define CREDSTRUCT cmsgcred
#define SCM_CREDTYPE SCM_CREDS
#elif defined(SCM_CREDENTIALS)
#define CREDSTRUCT ucred
#define SCM_CREDTYPE SCM_CREDENTIALS
#else
#error passing credentials is not supported!
#endif

#define RIGHTLEN CMSG_LEN(sizeof(int))
#define CREDSLEN CMSG_LEN(sizeof(struct CREDSTRUCT))
#define  CONTROL_MSG_LEN (RIGHTLEN + CREDSLEN)

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
		struct cmsghdr* cmp;
		cmp->cmsg_level = SOL_SOCKET;
		cmp->cmsg_type = SCM_RIGHTS;
		cmp->cmsg_len = RIGHTLEN;
		*(int*)CMSG_DATA(cmptr) = fd_to_send;
		cmp = CMSG_NXTHDR(&msg, cmp);
		cmp->cmsg_level = SOL_SOCKET;
		cmp->cmsg_type = SCM_CREDTYPE;
		cmp->cmsg_len = CREDSLEN;
#if defined(SCM_CREDENTIALS)
		struct CREDSTRUCT* credp = (struct CREDSTRUCT*)CMSG_DATA(cmp);
		credp->uid = geteuid();
		credp->gid = getegid();
		credp->pid = getpid();
#endif
		buf[1] = 0;
	}
	buf[0] = 0;
	if(sendmsg(fd, &msg, 0) != 2)
		return -1;
	return 0;
}

