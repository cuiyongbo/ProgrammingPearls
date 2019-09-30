#include "unp.h"

int listenFd, connFd;

void sig_urg(int signo)
{
    printf("SIGURG received\n");

    char buff[2048];
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

    const int on = 0;
    Setsockopt(listenFd, SOL_SOCKET, SO_OOBINLINE, &on, sizeof(on));

    const int recvBufferSize = 4096;
    Setsockopt(listenFd, SOL_SOCKET, SO_RCVBUF, &recvBufferSize, sizeof(recvBufferSize));

    connFd = Accept(listenFd, NULL, NULL);
    Signal(SIGURG, sig_urg);
    Fcntl(connFd, F_SETOWN, getpid());

    for(;;)
    {
        pause();
    }
}
