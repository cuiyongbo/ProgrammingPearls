#include "opend.h"
#include <poll.h>

#define NALLOC 10

static struct pollfd* grow_pollfd(struct pollfd* pfd, int* maxfd)
{
	int oldmax = *maxfd;
	int newmax = oldmax + NALLOC;
	if((pfd = realloc(pfd, newmax*sizeof(struct pollfd))) == NULL)
		err_sys("realloc error");
	for(int i=oldmax; i<newmax; i++)
	{
		pfd[i].fd = -1;
		pfd[i].events = POLL_IN;
		pfd[i].revents = 0;
	}
	*maxfd = newmax;
	return pfd;
}

void loop()
{
	struct pollfd* pfd = (struct pollfd*)malloc(NALLOC*sizeof(struct pollfd));
	if(pfd == NULL)
		err_sys("malloc error");
	for(int i=0; i<NALLOC; i++)
	{
		pfd[i].fd = -1;
		pfd[i].events = POLL_IN;
		pfd[i].revents = 0;
	}

	int listenfd = serv_listen(CS_OPEN);
	if(listenfd < 0)
		err_sys("serv_listen error");

	client_add(listenfd, 0);
	pfd[0].fd = listenfd;
	int numfd = 1;
	int maxfd = NALLOC;
	int clifd;
	uid_t uid;
	char buf[BUFSIZ];
	for(;;)
	{
		if(poll(pfd, numfd, -1) < 0)
			err_sys("poll error");
		if(pfd[0].revents & POLL_IN)
		{
			clifd = serv_accept(listenfd, &uid);
			if(clifd < 0)
				err_sys("serv_accept error");
			client_add(clifd, uid);
			if(numfd == maxfd)
				grow_pollfd(pfd, &maxfd);
			pfd[numfd].fd = clifd;
			pfd[numfd].events = POLL_IN;
			pfd[numfd].revents = 0;
			numfd++;
			err_ret("new connection: fd<%d>, uid<%ld>", clifd, (long)uid);
		}
		for(int i=1; i<numfd; i++)
		{
			clifd = pfd[i].fd;
			if(pfd[i].revents & POLL_IN)
			{
				int nread = read(clifd, buf, BUFSIZ);
				if(nread < 0)
				{
					err_sys("read error");
				}
				else if(nread > 0)
				{
					handle_request_02(buf, nread, clifd, clientInfos[i].uid);
				}
				else
				{
					goto hungup;
				}
			}
			else if(pfd[i].revents & POLL_HUP)
			{
hungup:
				err_ret("closed: fd<%d>, uid<%ld>", clifd, (long)uid);
				client_del(clifd);
				close(clifd);
				if(i < numfd-1)
				{
					pfd[i].fd = pfd[numfd-1].fd;
					pfd[i].events = pfd[numfd-1].events;
					pfd[i].revents = pfd[numfd-1].revents;
					i--;
				}
				numfd--;
			}
		}	
	}
}
