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
			err_sys("uptime server: accept error");
		}
		set_cloexec(clfd);
		pid_t pid = fork();
		if(pid < 0)
		{
			err_sys("uptime server: fork error");
		}
		else if(pid == 0)
		{
			if(dup2(clfd, STDOUT_FILENO) != STDOUT_FILENO ||
				dup2(clfd, STDERR_FILENO) != STDERR_FILENO)
			{
				err_quit("uptime server: unexpected error");
			}
			close(clfd);
			execl("/usr/bin/uptime", "uptime", (char*)0);
			err_msg("uptime server: unexpected returned from exec");
		}
		else
		{
			close(clfd);
			int status;
			waitpid(pid, &status, 0);
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
	hint.ai_socktype = SOCK_STREAM;
	hint.ai_flags = AI_CANONNAME;
	struct addrinfo* ailist;
	int err = getaddrinfo(host, argv[1], &hint, &ailist);
	if(err != 0)
	{
		err_quit("getaddrinfo error: %s\n", gai_strerror(err));
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

