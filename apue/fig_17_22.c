#include "opend.h"

void handle_request(char* buf, int nread, int fd)
{
	if(buf[nread-1] != 0)
	{
		snprintf(errmsg, BUFSIZ-1, "request not null terminated: %*.*s\n",
			nread, nread, buf);
		send_err(fd, -1, errmsg);
		return;
	}
	
	if(buf_args(buf, cli_args) < 0)
	{
		send_err(fd, -1, errmsg);
		return;
	}

	int newfd = open(pathname, oflag);
	if(newfd < 0)
	{
		snprintf(errmsg, BUFSIZ, "can't open %s: %s\n",
			pathname, strerror(errno));
		send_err(fd, -1, errmsg);
		return;
	}

	if(send_fd(fd, newfd) < 0)
		err_sys("send_fd error");

	close(newfd);
}
