#include "unp.h"

int main(int argc, char** argv)
{
    int listenFd = Socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(13);
    //servAddr.sin_port = htons(9999);

    Bind(listenFd, (const struct sockaddr*)&servAddr, sizeof(servAddr));

    Listen(listenFd, LISTEN_QUEUE_LEN);

    char buff[MAXLINE];
    while(1)
    {
        struct sockaddr_in cliAddr;
        socklen_t len = sizeof(cliAddr);
        int connfd = Accept(listenFd, (struct sockaddr*)&cliAddr, &len);

        printf("Connection from %s:%d\n",
            inet_ntop(AF_INET, &cliAddr.sin_addr, buff, sizeof(buff)),
            ntohs(cliAddr.sin_port));

        time_t tick = time(NULL);
        snprintf(buff, sizeof(buff), "%.24s\r\n", ctime(&tick));

        Write(connfd, buff, strlen(buff));

/*
        for(int i=0; i<strlen(buff); ++i)
        {
            write(connfd, buff+i, 1);
        }
*/
        Close(connfd);
    }

    return 0;
}
