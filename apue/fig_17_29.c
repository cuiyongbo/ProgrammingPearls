#include "opend.h"
#include <sys/select.h>

void loop()
{
	int listenfd = serv_listen(CS_OPEN);
	if(listenfd < 0)
		err_sys("serv_listen error");

	fd_set allset, rset;
	FD_ZERO(&allset);
	FD_SET(listenfd, &allset);
	int maxfd = listenfd;
	int maxi = -1;
	int clifd;
	uid_t uid;
	for(;;)
	{
		rset = allset;
		int n = select(maxfd+1, &rset, NULL, NULL, NULL);
		if(n < 0)
			err_sys("select error");
		
		if(FD_ISSET(listenfd, &rset))
		{
			if((clifd = serv_accept(listenfd, &uid)) < 0)
				err_sys("serv_accept error");
			int idx = client_add(clifd, uid);
			FD_SET(clifd, &allset);
			maxfd = max(maxfd, clifd);
			maxi = max(maxi, idx);
			err_ret("new connection: fd<%d>, uid<%ld>", clifd, (long)uid);
			continue;
		}
		char buf[BUFSIZ];
		for(int i=0; i<=maxi; i++)
		{
			if((clifd = clientInfos[i].fd) < 0)
				continue;
			if(FD_ISSET(clifd, &rset))
			{
				int nread = read(clifd, buf, BUFSIZ);
				if(nread < 0)
				{
					err_sys("read error");
				}
				else if(nread == 0)
				{
					err_ret("closed: fd<%d>, uid<%ld>", clifd, (long)uid);
					client_del(clifd);
					FD_CLR(clifd, &allset);
					close(clifd);
				}
				else
				{
					handle_request_02(buf, nread, clifd, clientInfos[i].uid);
				}
			}
		}
	}
}

