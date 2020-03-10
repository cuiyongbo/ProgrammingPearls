#include "apue.h"
#include <sys/socket.h>

#if defined(SCM_CREDS)
#define CREDSTRUCT cmsgcred
#define CR_UID cmcred_uid
#define SCM_CREDTYPE SCM_CREDS
#elif defined(SCM_CREDENTIALS)
#define CREDSTRUCT ucred
#define CR_UID uid
#define CREDOPT SO_PASSCRED
#define SCM_CREDTYPE SCM_CREDENTIALS
#else
#error passing credentials is not supported!
#endif

#define RIGHTLEN CMSG_LEN(sizeof(int))
#define CREDSLEN CMSG_LEN(sizeof(struct CREDSTRUCT))
#define  CONTROL_MSG_LEN (RIGHTLEN + CREDSLEN)

static struct cmsghdr* cmptr = NULL;

int recv_ufd(int fd, uid_t* uidptr, UnixDomainSocketUserFunc func)
{
	char buf[BUFSIZ];
	struct iovec iov[1];
	struct msghdr msg;

#if defined(CREDOPT)
		const int on = 1;
		if(setsockopt(fd, SOL_SOCKET, CREDOPT, *on, sizeof(int)) < 0)
		{
			err_ret("setsockopt error");
			return -1;
		}
#endif
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
			err_ret("recvmsg error");
			return -1;
		}
		else if(nr == 0)
		{
			err_ret("connection closed by server");
			return -1;
		}

		int newfd;
		struct CREDOPT* credp;
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
				
					for(struct cmsghdr* cmp=CMSG_FIRSTHDR(&msg);
						cmp != NULL; cmp = CMSG_NXTHDR(&msg, cmp))
					{
						if(cmp->cmsg_level != SOL_SOCKET)
							continue;
						switch(cmp->cmsg_type)
						{
							case SCM_RIGHTS:
								newfd = *(int*)CMSG_DATA(cmptr);
								break;
							case SCM_CREDTYPE:
								credp = (struct CREDSTRUCT*)CMSG_DATA(cmp);
								*uidptr = credp->CR_UID;
						}
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

