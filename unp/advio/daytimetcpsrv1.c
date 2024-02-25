#include "unp.h"
#include <sys/socket.h>
#include <arpa/inet.h>

int main(int argc, char** argv)
{
	int listenFd = socket(AF_INET, SOCK_STREAM, 0);
	if(listenFd < 0)
	{
		err_sys("socket error");
	}

	struct sockaddr_in servAddr;
	bzero(&servAddr, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	servAddr.sin_port = htons(13);

	if(bind(listenFd, (const struct sockaddr*)&servAddr, sizeof(servAddr)) < 0)
	{
		err_sys("bind error");
	}

	if(listen(listenFd, LISTEN_QUEUE_LEN) < 0)
	{
		err_sys("listen error");
	}

	char buff[MAXLINE];
	struct sockaddr_in cliAddr;
	while(1)
	{
		socklen_t len = sizeof(cliAddr);
		int connfd = accept(listenFd, (struct sockaddr*)&cliAddr, &len);
		printf("Connection from %s:%d\n",
			inet_ntop(AF_INET, &cliAddr.sin_addr, buff, sizeof(buff)),
			ntohs(cliAddr.sin_port));
		time_t tick = time(NULL);
		snprintf(buff, sizeof(buff), "%.24s\r\n", ctime(&tick));
		write(connfd, buff, strlen(buff));
		close(connfd);
	}

	return 0;
}
