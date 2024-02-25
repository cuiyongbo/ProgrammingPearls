#include "apue.h"
#include <netdb.h>
#include <sys/socket.h>

extern int connect_with_retry(int, int, int, const struct sockaddr*, socklen_t);

void print_uptime(int sockfd)
{
	int n;
	char buf[BUFSIZ];
	while((n=recv(sockfd, buf, BUFSIZ, 0)) > 0)
		write(STDOUT_FILENO, buf, n);
	if(n < 0)
		err_sys("recv error");

	close(sockfd);
}

int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		err_quit("Usage: %s host service\n", argv[0]);
	}	

	struct addrinfo hint;
	memset(&hint, 0, sizeof(struct addrinfo));
	hint.ai_socktype = SOCK_STREAM;
	struct addrinfo* ailist;
	int err = getaddrinfo(argv[1], argv[2], &hint, &ailist);
	if(err != 0)
	{
		err_quit("getaddrinfo error: %s\n", gai_strerror(err));
	}

	for(struct addrinfo* aip = ailist; aip != NULL; aip = aip->ai_next)	
	{
		int sockfd = connect_with_retry(aip->ai_family, SOCK_STREAM, 0,
			aip->ai_addr, aip->ai_addrlen);
		if(sockfd < 0)
		{
			err = errno;
		}
		else
		{
			print_uptime(sockfd);
			return 0;
		}		
	}
}

