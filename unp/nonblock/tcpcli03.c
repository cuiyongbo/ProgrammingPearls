#include "unp.h"

int main(int argc, char** argv)
{
    if(argc != 2)
        err_quit("Usage: %s server_ip", argv[0]);

    int sockFd = Socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(SERVER_PORT);
    Inet_pton(AF_INET, argv[1], &servAddr.sin_addr);
    Connect(sockFd, (SA*)&servAddr, sizeof(servAddr));

    // this causes an RST to be sent on a TCP socket when the connection is closed.
    struct linger ling;
    ling.l_onoff = 1;
    ling.l_linger = 0;
    Setsockopt(sockFd, SOL_SOCKET, SO_LINGER, &ling, sizeof(ling));
    return 0;
}
