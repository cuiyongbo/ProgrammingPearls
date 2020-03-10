#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        err_quit("Usage: %s server_ip", argv[0]);
    }

    int sockFd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockFd < 0)
    {
        err_sys("socket error");
    }

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(SERVER_PORT);
    if(inet_pton(AF_INET, argv[1], &servAddr.sin_addr) <= 0)
    {
        err_sys("inet_pton error for %s", argv[1]);
    }

    if(connect(sockFd, (SA*)&servAddr, sizeof(servAddr)) < 0)
    {
        err_sys("connect error");
    }

    str_cli(stdin, sockFd);

    return 0;
}
