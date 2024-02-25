#include "apue.h"
#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>

#define TIMEOUT 20

void sigalrm(int signo) {}

void print_uptime(int sockfd, struct addrinfo* aip)
{
	int n;
	char buf[BUFSIZ];
	buf[0] = 0;
	if(sendto(sockfd, buf, 1, 0, aip->ai_addr, aip->ai_addrlen) < 0)
		err_sys("sendto error");

	alarm(TIMEOUT);

	if((n=recvfrom(sockfd, buf, BUFSIZ, 0, NULL, NULL)) < 0)
	{
		if(errno != EINTR)
			alarm(0);
		err_sys("recvfrom error");
	}
	alarm(0);
	write(STDOUT_FILENO, buf, n);
	close(sockfd);
}

int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		err_quit("Usage: %s host service\n", argv[0]);
	}	

	struct sigaction sa;
	sa.sa_handler = sigalrm;
	sa.sa_flags = 0;
	sigemptyset(&sa.sa_mask);
	if(sigaction(SIGALRM, &sa, NULL)<0)
		err_sys("sigaction(SIGALRM)");

	struct addrinfo hint;
	memset(&hint, 0, sizeof(struct addrinfo));
	hint.ai_socktype = SOCK_DGRAM;
	struct addrinfo* ailist;
	int err = getaddrinfo(argv[1], argv[2], &hint, &ailist);
	if(err != 0)
	{
		err_quit("getaddrinfo error: %s\n", gai_strerror(err));
	}

	for(struct addrinfo* aip = ailist; aip != NULL; aip = aip->ai_next)	
	{
		int sockfd = socket(aip->ai_family, SOCK_DGRAM, 0);
		if(sockfd < 0)
		{
			err = errno;
			err_msg("socket");
		}
		else
		{
			print_uptime(sockfd, aip);
			return 0;
		}		
	}
	err_sys("cannot contact %s<%s>", argv[1], argv[2]);
}

