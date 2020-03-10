#include "apue.h"
#include <netdb.h>
#include <syslog.h>
#include <sys/socket.h>

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 256
#endif

extern int initServer(int, const struct sockaddr*, socklen_t, int);

void serve(int sockfd)
{
	set_cloexec(sockfd);

	char buf[BUFSIZ];
	char addrBuf[BUFSIZ];
	for(;;)
	{
		struct sockaddr* addr = (struct sockaddr*)addrBuf;
		socklen_t addrLen = sizeof(addrBuf);
		int n = recvfrom(sockfd, buf, BUFSIZ, 0, addr, &addrLen);
		if(n < 0)
			err_sys("recv error");
		FILE* fp = popen("/usr/bin/uptime", "r");
		if(fp == NULL)
		{
			sprintf(buf, "error: %s\n", strerror(errno));
			sendto(sockfd, buf, strlen(buf), 0, addr, addrLen);	
		}
		else
		{
			if(fgets(buf, BUFSIZ, fp) != NULL)
				sendto(sockfd, buf, strlen(buf), 0, addr, addrLen);	
			pclose(fp);
		}
	}
}

int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		err_quit("Usage: %s service\n", argv[0])
		return 1;
	}	

	int n;
	if((n = sysconf(_SC_HOST_NAME_MAX)) < 0)
		n = HOST_NAME_MAX;
	char* host = (char*)malloc(n);
	if(host == NULL)
		err_sys("malloc error");
	if(gethostname(host, n) < 0)
		err_sys("gethostname error");
	printf("host: %s\n", host);
	//daemonize(argv[0]);	

	struct addrinfo hint;
	memset(&hint, 0, sizeof(struct addrinfo));
	hint.ai_socktype = SOCK_DGRAM;
	hint.ai_flags = AI_CANONNAME;
	struct addrinfo* ailist;
	int err = getaddrinfo(host, argv[1], &hint, &ailist);
	if(err != 0)
	{
		err_quit("getaddrinfo error: %s\n", gai_strerror(err));
	}

	for(struct addrinfo* aip = ailist; aip != NULL; aip = aip->ai_next)	
	{
		int sockfd = initServer(SOCK_DGRAM, aip->ai_addr, aip->ai_addrlen, 0);
		if(sockfd >= 0)
		{
			serve(sockfd);
			return 0;
		}		
	}
}

