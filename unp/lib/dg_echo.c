#include    "unp.h"

void dg_echo(int sockfd, SA* pcliaddr, socklen_t clilen)
{
    char msg[MAXLINE];
    while(1)
    {
        socklen_t len = clilen;
        int n = Recvfrom(sockfd, msg, MAXLINE, 0, pcliaddr, &len);
        Sendto(sockfd, msg, n, 0, pcliaddr, len);
    }
}
