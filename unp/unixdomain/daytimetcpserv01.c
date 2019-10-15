#include    "unp.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        err_quit("usage: %s <service or port#>", argv[0]);
    }

    int listenfd = Tcp_listen(NULL, argv[1], NULL);

    char buff[MAXLINE];
    struct sockaddr_storage cliaddr;
    for ( ; ; )
    {
        socklen_t len = sizeof(cliaddr);
        int connfd = Accept(listenfd, (SA *)&cliaddr, &len);
        printf("connection from %s\n", Sock_ntop_host((SA*)&cliaddr, len));

        time_t ticks = time(NULL);
        snprintf(buff, sizeof(buff), "%.24s\r\n", ctime(&ticks));

        size_t n = strlen(buff);
        for(int i=0; i<n; ++i)
        {
            write(connfd, &buff[i], 1);
        }

        //Write(connfd, buff, strlen(buff));

        Close(connfd);
    }
}
