#include "web.h"

void home_page(const char* host, const char* fname)
{
    int fd = Tcp_connect(host, SERV);

    char line[MAXLINE];
    int n = snprintf(line, sizeof(line), GET_CMD, fname);
    Writen(fd, line, n);

    while(1)
    {
        if((n=Read(fd, line, MAXLINE)) == 0)
            break;

        printf("read %d bytes of home page\n", n);
    }

    printf("eof on home page\n");
    Close(fd);
}
