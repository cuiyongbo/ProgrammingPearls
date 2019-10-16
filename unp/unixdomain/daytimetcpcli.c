#include    "unp.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        err_quit("Usage: %s <hostname/IPaddress> <service/port#>", argv[0]);
    }

    int sockfd = Tcp_connect(argv[1], argv[2]);

    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    Getpeername(sockfd, (SA *)&ss, &len);
    printf("connected to %s\n", Sock_ntop_host((SA*)&ss, len));

    sleep(5);

    ssize_t n = 0;
    char recvline[MAXLINE + 1];
    while((n = Read(sockfd, recvline, MAXLINE)) > 0)
    {
        printf("receive %d bytes\n", (int)n);
        recvline[n] = 0;    /* null terminate */
        Fputs(recvline, stdout);
    }
    exit(0);
}
