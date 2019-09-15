#include "apue.h"
#include <sys/socket.h>
#include <arpa/inet.h>

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		printf("Usage: %s server_ip\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	int sockFd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockFd < 0)
	{
		err_sys("socket error");
	}

	struct sockaddr_in servAddr;
	bzero(&servAddr, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(13);
	if(inet_pton(AF_INET, argv[1], &servAddr.sin_addr) <= 0)
	{
		err_sys("inet_pton error for %s", argv[1]);
	}

	if(connect(sockFd, (const struct sockaddr*)&servAddr, sizeof(servAddr)) < 0)
	{
		err_sys("connect error");
	}

	int n = 0;
	char buff[MAXLINE];
	struct sockaddr_in cliAddr;

	while((n = read(sockFd, buff, sizeof(MAXLINE))) > 0)
	{
		buff[n] = 0;
		printf("%s", buff);
	}

	if(n < 0)
	{
		err_msg("read error");
	}

	close(sockFd);
	return 0;
}
