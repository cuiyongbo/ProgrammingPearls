#include    "unp.h"

int main(int argc, char **argv)
{
    if(argc < 2 || argc > 3)
        err_quit("usage: %s [ <host> ] <service or port>", argv[0]);

    daemon_inetd(argv[0], 0);

    socklen_t addrlen = sizeof(struct sockaddr_storage);
    SA* cliaddr = (SA*)Malloc(addrlen);

    Getpeername(0, cliaddr, &addrlen);
    err_msg("connection from %s", Sock_ntop_host(cliaddr, addrlen));

    char buff[MAXLINE];
    time_t ticks = time(NULL);
    snprintf(buff, sizeof(buff), "%.24s\r\n", ctime(&ticks));
    Write(0, buff, strlen(buff));
    Close(0);
    return 0;
}
