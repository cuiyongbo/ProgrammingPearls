#include "cli_serv.h"

int main(int argc, char* argv[])
{
	if (argc != 2)
		err_quit("usage: %s <IP address of server>", argv[0]);
	
	int sockFd = socket(PF_INET, SOCK_STREAM, 0);
	if(sockFd < 0)
		err_sys("socket error");

	struct sockaddr_in serv;
	memset(&serv, 0, sizeof(serv));
	serv.sin_family = AF_INET;
	serv.sin_addr.s_addr = inet_addr(argv[1]);
	serv.sin_port = htons(TCP_SERVER_PORT);
	
	if(connect(sockFd, (SA)&serv, sizeof(serv)) != 0)
		err_sys("connect error");

	char request[REQUEST_SIZE], reply[REPLY_SIZE];
	strncpy(request, "Hello server!", REQUEST_SIZE);

	size_t requestSize = strlen(request) + 1;
	if (write(sockFd, request, requestSize) != (ssize_t)requestSize)
		err_sys("write error");

	shutdown(sockFd, SHUT_WR); // send FIN to server

	int bytesReceived = read_stream(sockFd, reply, REPLY_SIZE);
	if (bytesReceived < 0)
		err_sys("read_stream error");
	
	printf("Receive %d bytes: %s\n", bytesReceived, reply);

	return 0;
}

