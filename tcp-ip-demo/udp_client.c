#include "cli_serv.h"

int main(int argc, char* argv[])
{
	if (argc != 2)
		err_quit("usage: udpcli <IP address of server>");
	
	int sockFd = socket(PF_INET, SOCK_DGRAM, 0);
	if(sockFd < 0)
		err_sys("socket error");

	struct sockaddr_in serv;
	memset(&serv, 0, sizeof(serv));
	serv.sin_family = AF_INET;
	serv.sin_addr.s_addr = inet_addr(argv[1]);
	serv.sin_port = htons(UDP_SERVER_PORT);
	
	char request[REQUEST_SIZE], reply[REPLY_SIZE];
	strncpy(request, "Hello server!", REQUEST_SIZE);

	size_t requestSize = strlen(request) + 1;
	if (sendto(sockFd, request, requestSize, 0, 
					(SA)&serv, sizeof(serv)) != (ssize_t)requestSize)
		err_sys("sendto error");

	ssize_t bytesReceived = recvfrom(sockFd, reply, REPLY_SIZE, 0, (SA)NULL, (socklen_t*)NULL);
	if (bytesReceived < 0)
		err_sys("recvfrom error");
	
	printf("Receive %d bytes: %s\n", bytesReceived, reply);

	return 0;
}

