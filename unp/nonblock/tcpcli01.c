#include    "unp.h"


int main(int argc, char** argv)
{
    if(argc != 2)
        err_quit("Usage: %s <IP_ADDRESS>", argv[0]);

    int sockFd = Socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(7);
    Inet_pton(AF_INET, argv[1], &servAddr.sin_addr);
    Connect(sockFd, (SA*)&servAddr, sizeof(servAddr));

    str_cli(stdin, sockFd);

    return 0;
}
