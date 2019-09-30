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

    const int on = 0;
    Setsockopt(listenFd, SOL_SOCKET, SO_OOBINLINE, &on, sizeof(on));

    int connFd = Accept(listenFd, NULL, NULL);
    Fcntl(connFd, F_SETOWN, getpid());

    int n = 0;
    char buff[MAXLINE];
    fd_set rset, xset;
    FD_ZERO(&rset);
    FD_ZERO(&xset);
    for(;;)
    {
        FD_SET(connFd, &rset);
        FD_SET(connFd, &xset);

        Select(connFd+1, &rset, NULL, &xset, NULL);

        if(FD_ISSET(connFd, &xset))
        {
            printf("SIGURG received\n");
            n = Recv(connFd, buff, sizeof(buff)-1, MSG_OOB);
            buff[n] = 0;
            printf("read %d OOB byte: %s\n", n, buff);
        }

        if(FD_ISSET(connFd, &rset))
        {
            if((n = Read(connFd, buff, MAXLINE)) == 0)
                err_quit("EOF received\n");

            buff[n] = 0;
            printf("read %d bytes: %s\n", n, buff);
        }
    }
}
