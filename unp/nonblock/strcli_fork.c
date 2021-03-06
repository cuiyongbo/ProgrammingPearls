#include "unp.h"

void str_cli(FILE* fp, int sockFd)
{
    char recvBuff[MAXLINE], sendBuff[MAXLINE];
    pid_t pid = Fork();
    if(pid == 0)
    {
        ssize_t n;
        while((n=read(sockFd, recvBuff, MAXLINE)) > 0)
        {
            recvBuff[n] = 0;
            Fputs(recvBuff, stdout);
        }
        kill(getppid(), SIGTERM);
        exit(0);
    }
    else
    {
        while(Fgets(sendBuff, MAXLINE, stdin) != NULL)
        {
            Writen(sockFd, sendBuff, strlen(sendBuff));
        }

        Shutdown(sockFd, SHUT_WR);
        pause();
    }
}
