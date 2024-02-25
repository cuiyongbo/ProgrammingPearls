#include "unp.h"


int main(int argc, char** argv)
{
    int listenFd;
    if(argc == 2)
        listenFd = Tcp_listen(NULL, argv[1], NULL);
    else if(argc == 3)
        listenFd = Tcp_listen(argv[1], argv[2], NULL);
    else
        err_quit("Usage: %s [host] <port#>", argv[0]);

    const int on = 1;
    Setsockopt(listenFd, SOL_SOCKET, SO_OOBINLINE, &on, sizeof(on));

    int connFd = Accept(listenFd, NULL, NULL);
    sleep(5);

    int n = 0;
    char buff[MAXLINE];
    for(;;)
    {
        if(sockatmark(connFd))
            printf("at OOB mark");

        if((n = Read(connFd, buff, MAXLINE)) == 0)
            err_quit("EOF received\n");

        buff[n] = 0;
        printf("read %d bytes: %s\n", n, buff);
    }
}
