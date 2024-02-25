#include    "unp.h"

int main(int argc, char **argv)
{
    int listenfd = Socket(AF_LOCAL, SOCK_STREAM, 0);

    unlink(UNIXSTR_PATH);

    struct sockaddr_un servaddr, cliaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sun_family = AF_LOCAL;
    strcpy(servaddr.sun_path, UNIXSTR_PATH);
    Bind(listenfd, (SA*)&servaddr, sizeof(servaddr));

    listen(listenfd, LISTEN_QUEUE_LEN);

    char buff[MAXLINE];
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
            send(connfd, &buff[i], 1, MSG_EOR);
        }

        Close(connfd);
    }
}
