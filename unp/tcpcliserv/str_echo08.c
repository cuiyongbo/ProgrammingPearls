#include "unp.h"

void str_echo(int sockFd)
{
    long arg1, arg2;
    char line[MAXLINE];
    for(;;)
    {
        ssize_t n = Readline(sockFd, line, MAXLINE);
        if(n == 0) return;

        if(sscanf(line, "%ld %ld", &arg1, &arg2) == 2)
            snprintf(line, sizeof(line), "%ld\n", arg1+arg2);
        else
            snprintf(line, sizeof(line), "input error\n");

        Writen(sockFd, line, strlen(line));
    }
}
