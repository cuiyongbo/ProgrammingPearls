#include "apue.h"
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>

void printFamily(const struct addrinfo* aip)
{
	printf(" address family ");
	switch(aip->ai_family)
	{
	case AF_INET:
		printf("inet");
		break;
	case AF_INET6:
		printf("inet6");
		break;
	case AF_LOCAL:
		printf("local");
		break;
	case AF_UNSPEC:
		printf("unspecified");
		break;
	default:
		printf("unknown (%d)", aip->ai_family);
	}
}

void printType(const struct addrinfo* aip)
{
	printf(" type ");
	switch(aip->ai_socktype)
	{
	case SOCK_STREAM:
		printf("stream");
		break;
	case SOCK_DGRAM:
		printf("datagram");
		break;
	case SOCK_SEQPACKET:
		printf("seqpacket");
		break;
	case SOCK_RAW:
		printf("raw");
		break;
	default:
		printf("unknown (%d)", aip->ai_socktype);
	}
}

void printProtocol(const struct addrinfo* aip)
{
	printf(" protocol ");
	switch(aip->ai_protocol)
	{
	case 0:
		printf("default");
		break;
	case IPPROTO_TCP:
		printf("TCP");
		break;
	case IPPROTO_UDP:
		printf("UDP");
		break;
	case IPPROTO_RAW:
		printf("raw");
		break;
	default:
		printf("unknown (%d)", aip->ai_protocol);
	}
}

void printFlags(const struct addrinfo* aip)
{
	printf("flags");
	if(aip->ai_flags == 0)
	{
		printf(" 0");
	}
	else
	{
		if(aip->ai_flags & AI_PASSIVE)
		{
			printf(" passive");
		}
		if(aip->ai_flags & AI_CANONNAME)
		{
			printf(" canon");
		}
		if(aip->ai_flags & AI_NUMERICHOST)
		{
			printf(" numhost");
		}
		if(aip->ai_flags & AI_NUMERICSERV)
		{
			printf(" numserv");
		}
		if(aip->ai_flags & AI_V4MAPPED)
		{
			printf(" v4mapped");
		}
		if(aip->ai_flags & AI_ALL)
		{
			printf(" all");
		}
	}
}

int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		printf("Usage: %s nodename service\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	struct addrinfo hint;
	memset(&hint, 0, sizeof(struct addrinfo));
	hint.ai_flags = AI_CANONNAME;
	struct addrinfo* ailist;
	int err = getaddrinfo(argv[1], argv[2], &hint, &ailist);
	if(err != 0)
	{
		printf("getaddrinfo error: %s\n", gai_strerror(err));
		exit(EXIT_FAILURE);
	}

	for(struct addrinfo* aip=ailist; aip != NULL; aip = aip->ai_next)
	{
		printFlags(aip);
		printFamily(aip);
		printType(aip);
		printProtocol(aip);
		printf("\n\thost %s", aip->ai_canonname ? aip->ai_canonname : "-");
		if(aip->ai_family == AF_INET)
		{
			char abuf[INET_ADDRSTRLEN];
			struct sockaddr_in* sinp = (struct sockaddr_in*)aip->ai_addr;
			const char* addr = inet_ntop(AF_INET, &sinp->sin_addr, abuf, INET_ADDRSTRLEN);
			printf(" address %s", addr ? addr : "unknown");
			printf(" port %d", ntohs(sinp->sin_port));
		}
		printf("\n");
	}
	exit(0);
}

