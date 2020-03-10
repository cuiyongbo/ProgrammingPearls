#include    "unp.h"

int main(int argc, char **argv)
{
    if(argc < 2 || argc > 3)
        err_quit("usage: %s [ <host> ] <service or port>", argv[0]);

    daemon_init(argv[0], 0);

    int listenfd;
    socklen_t addrlen;
    if (argc == 2)
        listenfd = Tcp_listen(NULL, argv[1], &addrlen);
    else
        listenfd = Tcp_listen(argv[1], argv[2], &addrlen);

    SA* cliaddr = (SA*)Malloc(addrlen);

    char buff[MAXLINE];
    for ( ; ; ) {
        socklen_t len = addrlen;
        int connfd = Accept(listenfd, cliaddr, &len);
        err_msg("connection from %s\n", Sock_ntop_host(cliaddr, len));

        time_t ticks = time(NULL);
        snprintf(buff, sizeof(buff), "%.24s\r\n", ctime(&ticks));
        Write(connfd, buff, strlen(buff));
        Close(connfd);
    }
    return 0;
}
