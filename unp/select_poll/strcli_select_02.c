#include "unp.h"

void str_cli(FILE* fp, int sockFd)
{
    fd_set rset;
    int fd = fileno(fp);
    int maxFd = max(fd, sockFd) + 1;
    char buff[MAXLINE];
    int stdin_eof = 0;
    while(1)
    {
        FD_ZERO(&rset);

        if(stdin_eof == 0)
            FD_SET(fd, &rset);

        FD_SET(sockFd, &rset);
        Select(maxFd, &rset, NULL, NULL, NULL);

        if(FD_ISSET(sockFd, &rset))
        {
            ssize_t n = Read(sockFd, buff, MAXLINE);
            if(n == 0)
            {
                if(stdin_eof == 1)
                    return;
                else
                    err_quit("str_cli: server terminated prematurely");
            }

            Write(fileno(stdout), buff, n);
        }

        if(FD_ISSET(fd, &rset))
        {
            ssize_t n = Read(fd, buff, MAXLINE);
            if(n == 0)
            {
                stdin_eof = 1;
                Shutdown(sockFd, SHUT_WR);
                FD_CLR(fd, &rset);
                continue;
            }

            Write(sockFd, buff, n);
        }
    }
}
