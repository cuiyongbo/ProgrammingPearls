#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        err_quit("Usage: %s server_ip", argv[0]);
    }

    int sockFds[5];
    struct sockaddr_in servAddr;
    for(int i=0; i<element_of(sockFds); ++i)
    {
        sockFds[i] = socket(AF_INET, SOCK_STREAM, 0);
        if(sockFds[i] < 0)
        {
            err_sys("socket error");
        }

        bzero(&servAddr, sizeof(servAddr));
        servAddr.sin_family = AF_INET;
        servAddr.sin_port = htons(SERVER_PORT);
        if(inet_pton(AF_INET, argv[1], &servAddr.sin_addr) <= 0)
        {
            err_sys("inet_pton error for %s", argv[1]);
        }

        if(connect(sockFds[i], (SA*)&servAddr, sizeof(servAddr)) < 0)
        {
            err_sys("connect error");
        }

    }

    str_cli(stdin, sockFds[0]);

    return 0;
}
