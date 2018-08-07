#include "cli_serv.h"

int main(int argc, char* argv[])
{
	UNUSED_VAR(argc);
	UNUSED_VAR(argv);

	int listenFd = socket(PF_INET, SOCK_STREAM, 0);
	if(listenFd < 0)
		err_sys("socket error");
	
	int on = 1;
	if(setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) == -1)
		err_sys("setsockopt error");

	struct sockaddr_in serv;
	memset(&serv, 0, sizeof(serv));
	serv.sin_family = AF_INET;
	serv.sin_addr.s_addr = htonl(INADDR_ANY);
	serv.sin_port = htons(TCP_SERVER_PORT);
	if(bind(listenFd, (SA)&serv, sizeof(serv)) < 0)
		err_sys("bind error");

	if(listen(listenFd, SOMAXCONN) < 0)
		err_sys("listen error");

	char request[REQUEST_SIZE], reply[REPLY_SIZE];
	strncpy(reply, "Hello client!", REPLY_SIZE);

	for(;;)
	{
		struct sockaddr_in cli;
		socklen_t cliLen = sizeof(cli); 
		int clientFd = accept(listenFd, (SA)&cli, &cliLen);	
		if (clientFd < 0)
			err_sys("accept error");

		int bytesReceived = read_stream(clientFd, request, REQUEST_SIZE);
		if (bytesReceived < 0)
			err_sys("read_stream error");

		printf("Receive %d bytes: %s\n", bytesReceived, request);

		size_t replySize = strlen(reply) + 1;
		if (write(clientFd, reply, replySize) != (ssize_t)replySize)
			err_sys("write error");
		
		close(clientFd);		
	}
	
	return 0;
}

