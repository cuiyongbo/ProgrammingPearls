#include "unp.h"

int main(int argc, char** argv)
{
    int listenFd = Socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in servAddr;
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(SERVER_PORT);
    Bind(listenFd, (SA*)&servAddr, sizeof(servAddr));

    Listen(listenFd, LISTEN_QUEUE_LEN);

    Signal(SIGCHLD, sig_child);

    int clientFds[FD_SETSIZE];
    for (int i = 0; i < FD_SETSIZE; ++i)
    {
        clientFds[i] = -1;
    }

    int maxFd = listenFd, maxi=-1;
    fd_set allset;
    FD_ZERO(&allset);
    FD_SET(listenFd, &allset);

    char buff[MAXLINE];
    while(1)
    {
        fd_set rset = allset;
        int nready = Select(maxFd+1, &rset, NULL, NULL, NULL);

        if(FD_ISSET(listenFd, &rset))
        {
            printf("Listening socket readable\n");
            sleep(5);

            struct sockaddr_in cliAddr;
            socklen_t cliLen = sizeof(cliAddr);
            int cliFd = Accept(listenFd, (SA*)&cliAddr, &cliLen);
            printf("Connection from %s:%d\n",
                inet_ntop(AF_INET, &cliAddr.sin_addr, buff, MAXLINE),
                ntohs(cliAddr.sin_port));

            int isFull = 1;
            for (int i = 0; i < FD_SETSIZE; ++i)
            {
                if (clientFds[i] == -1)
                {
                    clientFds[i] = cliFd;
                    maxi = max(maxi, i);
                    isFull = 0;
                    break;
                }
            }

            if(isFull)
                err_quit("server is busy now");

            FD_SET(cliFd, &allset);
            maxFd = max(maxFd, cliFd);

            if(--nready == 0)
                continue;
        }

        for (int i = 0; i <= maxi; ++i)
        {
            if(clientFds[i] == -1) continue;
            if(FD_ISSET(clientFds[i], &rset))
            {
                int n = Read(clientFds[i], buff, MAXLINE);
                if(n == 0)
                {
                    close(clientFds[i]);
                    FD_CLR(clientFds[i], &allset);
                    clientFds[i] = -1;
                }
                else
                {
                    Writen(clientFds[i], buff, n);
                }

                if(--nready == 0)
                    break;
            }
        }
    }
}
