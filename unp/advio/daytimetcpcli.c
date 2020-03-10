#include "unp.h"

int main(int argc, char** argv)
{
	if(argc != 2)
		err_quit("Usage: %s server_ip\n", argv[0]);

	int sockFd = Socket(AF_INET, SOCK_STREAM, 0);

	struct sockaddr_in servAddr;
	bzero(&servAddr, sizeof(servAddr));
	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(13);
	if(inet_pton(AF_INET, argv[1], &servAddr.sin_addr) <= 0)
	{
		err_sys("inet_pton error for %s", argv[1]);
	}

	Connect(sockFd, (const SA*)&servAddr, sizeof(servAddr));

	int n = 0;
	char buff[MAXLINE];
	while((n = read(sockFd, buff, sizeof(MAXLINE))) > 0)
	{
		buff[n] = 0;
		printf("%s", buff);
	}

	if(n < 0)
	{
		err_msg("read error");
	}

	Close(sockFd);
	return 0;
}
