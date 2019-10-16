#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        err_quit("Usage: %s <hostname> <service>", argv[0]);
    }

    socklen_t salen;
    struct sockaddr* sa;
    int sockfd = Udpclient(argv[1], argv[2], &sa, &salen);

    dg_cli(stdin, sockfd, sa, salen);

    exit(0);
}
