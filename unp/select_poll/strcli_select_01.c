#include "unp.h"

void str_cli(FILE* fp, int sockFd)
{
    fd_set rset;
    int fd = fileno(fp);
    int maxFd = max(fd, sockFd) + 1;
    char sendBuf[MAXLINE], recvBuf[MAXLINE];
    while(1)
    {
        FD_ZERO(&rset);
        FD_SET(fd, &rset);
        FD_SET(sockFd, &rset);
        Select(maxFd, &rset, NULL, NULL, NULL);

        if(FD_ISSET(sockFd, &rset))
        {
            ssize_t n = read(sockFd, recvBuf, sizeof(recvBuf));
            if(n < 0)
            {
                err_sys("read error");
            }
            else if(n == 0)
            {
                err_quit("str_cli: server terminated prematurely");
            }
            else
            {
                recvBuf[n] = 0;
                fputs(recvBuf, stdout);
            }
        }

        if(FD_ISSET(fd, &rset))
        {
            if(fgets(sendBuf, sizeof(sendBuf), fp) == NULL)
                return;

            write(sockFd, sendBuf, strlen(sendBuf));
        }
    }
}
