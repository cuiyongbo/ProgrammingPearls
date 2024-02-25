#include    "unp.h"

int main(int argc, char **argv)
{
    int listenfd = 0;
    socklen_t addrlen = 0;
    if (argc == 2)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else if (argc == 3)
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);
    else
        err_quit("usage: %s [ <host> ] <service or port>", argv[0]);

    char buff[MAXLINE];
    struct sockaddr_storage cliaddr;
    for ( ; ; )
    {
        addrlen = sizeof(cliaddr);
        int connfd = Accept(listenfd, (SA *)&cliaddr, &addrlen);
        printf("connection from %s\n", Sock_ntop_host((SA *)&cliaddr, addrlen));

        time_t ticks = time(NULL);
        snprintf(buff, sizeof(buff), "%.24s\r\n", ctime(&ticks));
        Write(connfd, buff, strlen(buff));
        Close(connfd);
    }
}
