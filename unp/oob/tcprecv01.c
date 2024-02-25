#include "unp.h"

int listenFd, connFd;

void sig_urg(int signo)
{
    printf("SIGURG received\n");

    char buff[MAXLINE];
    int n = Recv(connFd, buff, sizeof(buff)-1, MSG_OOB);
    buff[n] = 0;
    printf("read %d OOB byte: %s\n", n, buff);
}

int main(int argc, char** argv)
{
    if(argc == 2)
        listenFd = Tcp_listen(NULL, argv[1], NULL);
    else if(argc == 3)
        listenFd = Tcp_listen(argv[1], argv[2], NULL);
    else
        err_quit("Usage: %s [host] <port#>", argv[0]);

    const int on = 1;
    Setsockopt(listenFd, SOL_SOCKET, SO_OOBINLINE, &on, sizeof(on));

    connFd = Accept(listenFd, NULL, NULL);
    Signal(SIGURG, sig_urg);
    Fcntl(connFd, F_SETOWN, getpid());

    int n = 0;
    char buff[MAXLINE];
    for(;;)
    {
        if((n = Read(connFd, buff, MAXLINE)) == 0)
            err_quit("EOF received\n");

        buff[n] = 0;
        printf("read %d bytes: %s\n", n, buff);
    }
}
