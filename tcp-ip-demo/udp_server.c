#include "cli_serv.h"

int main(int argc, char* argv[])
{
	UNUSED_VAR(argc);
	UNUSED_VAR(argv);

	int sockFd = socket(PF_INET, SOCK_DGRAM, 0);
	if(sockFd < 0)
		err_sys("socket error");

	struct sockaddr_in serv;
	memset(&serv, 0, sizeof(serv));
	serv.sin_family = AF_INET;
	serv.sin_addr.s_addr = htonl(INADDR_ANY);
	serv.sin_port = htons(UDP_SERVER_PORT);
	
	if(bind(sockFd, (SA)&serv, sizeof(serv)) < 0)
		err_sys("bind error");

	char request[REQUEST_SIZE], reply[REPLY_SIZE];
	strncpy(reply, "Hello client!", REPLY_SIZE);

	for(;;)
	{
		struct sockaddr_in cli;
		socklen_t cliLen = sizeof(cli); 
		ssize_t bytesReceived = recvfrom(sockFd, request, REQUEST_SIZE, 0, (SA)&cli, &cliLen);
		if (bytesReceived < 0)
			err_sys("recvfrom error");

		printf("Receive %d bytes: %s\n", bytesReceived, request);

		size_t replySize = strlen(reply) + 1;
		if (sendto(sockFd, reply, replySize, 0, (SA)&cli, cliLen) != (ssize_t)replySize)
			err_sys("sendto error");		
	}
	
	return 0;
}

