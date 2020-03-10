#include "unp.h"

void str_echo(int sockFd)
{
    ssize_t n;
    char buf[MAXLINE];

again:
    while((n=read(sockFd, buf, sizeof(buf))) > 0)
    {
        buf[n] = 0;
        write(sockFd, buf, n);
    }

    if(n<0 && errno == EINTR)
        goto again;
    else if(n<0)
        err_sys("str_echo error");
}
