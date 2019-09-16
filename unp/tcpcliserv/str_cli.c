#include "unp.h"

void str_cli(FILE* fp, int sockFd)
{
    char sendBuf[MAXLINE], recvBuf[MAXLINE];
    while(fgets(sendBuf, sizeof(sendBuf), fp) != NULL)
    {
        write(sockFd, sendBuf, strlen(sendBuf));
		//memset(sendBuf, 0, sizeof(sendBuf));

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
            fputs(recvBuf, stdout);
        }
    }
}
