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

    struct pollfd pfd;
    pfd.events = POLLIN;
    pfd.fd = Accept(listenFd, NULL, NULL);

    int n = 0;
    char buff[MAXLINE];
    int justReadOOB = 0;

    for(;;)
    {
        if(justReadOOB == 0)
            pfd.events |= POLLPRI;

        Poll(&pfd, 1, -1);

        if(pfd.revents & POLLPRI)
        {
            n = Recv(pfd.fd, buff, sizeof(buff)-1, MSG_OOB);
            buff[n] = 0;
            printf("read %d OOB byte: %s\n", n, buff);

            justReadOOB = 1;
            pfd.events &= ~POLLPRI;
        }

        if(pfd.revents & POLLIN)
        {
            if((n = Read(pfd.fd, buff, MAXLINE)) == 0)
                err_quit("EOF received\n");

            buff[n] = 0;
            printf("read %d bytes: %s\n", n, buff);

            justReadOOB = 0;
        }
    }
}
