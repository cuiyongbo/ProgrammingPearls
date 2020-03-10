#include "apue.h"
#include <netdb.h>
#include <syslog.h>
#include <sys/socket.h>

#define QUEUE_LENGTH 10

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 256
#endif

extern int initServer(int, const struct sockaddr*, socklen_t, int);

void serve(int sockfd)
{
	set_cloexec(sockfd);
	for(;;)
	{
		int clfd = accept(sockfd, NULL, NULL);
		if(clfd < 0)
		{
			syslog(LOG_ERR, "uptime server: accept error: %s\n", strerror(errno));
			exit(1);
		}
		set_cloexec(clfd);
		char buf[BUFSIZ];
		FILE* fp = popen("/usr/bin/uptime", "r");
		if(fp == NULL)
		{
			sprintf(buf, "error: %s\n", strerror(errno));
			send(clfd, buf, strlen(buf), 0);	
		}
		else
		{
			while(fgets(buf, BUFSIZ, fp) != NULL)
				send(clfd, buf, strlen(buf), 0);
			pclose(fp);
		}
		close(clfd);
	}
}

int main(int argc, char* argv[])
{
	if(argc != 1)
	{
		printf("Usage: %s\n", argv[0]);
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
	daemonize(argv[0]);	

	struct addrinfo hint;
	memset(&hint, 0, sizeof(struct addrinfo));
	hint.ai_socktype = SOCK_STREAM;
	hint.ai_flags = AI_CANONNAME;
	struct addrinfo* ailist;
	int err = getaddrinfo(host, "ruptime", &hint, &ailist);
	if(err != 0)
	{
		printf("getaddrinfo error: %s\n", gai_strerror(err));
		exit(1);
	}

	for(struct addrinfo* aip = ailist; aip != NULL; aip = aip->ai_next)	
	{
		int sockfd = initServer(SOCK_STREAM, aip->ai_addr, aip->ai_addrlen, QUEUE_LENGTH);
		if(sockfd >= 0)
		{
			serve(sockfd);
			return 0;
		}		
	}
}

