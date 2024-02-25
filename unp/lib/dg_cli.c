#include "unp.h"

void dg_cli(FILE* fp, int sockfd, const SA* pservaddr, socklen_t servlen)
{
    char sendBuf[MAXLINE], recvBuf[MAXLINE+1];
    while(Fgets(sendBuf, MAXLINE, fp) != NULL)
    {
        Sendto(sockfd, sendBuf, strlen(sendBuf), 0, pservaddr, servlen);

        int n = Recvfrom(sockfd, recvBuf, MAXLINE, 0, NULL, NULL);
        recvBuf[n] = 0;
        Fputs(recvBuf, stdout);
    }
}
