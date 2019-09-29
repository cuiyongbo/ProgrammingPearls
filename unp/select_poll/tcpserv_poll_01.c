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

    int maxi = 0;
    char buff[MAXLINE];

    long maxOpenFileCount = sysconf(_SC_OPEN_MAX);
    if(maxOpenFileCount < 0)
        err_sys("sysconf error");

    struct pollfd* clients = (struct pollfd*)Malloc(maxOpenFileCount * sizeof(struct pollfd));
    clients[0].fd = listenFd;
    clients[0].events = POLLIN;
    for (int i = 1; i < maxOpenFileCount; ++i)
    {
        clients[i].fd = -1;
    }

    while(1)
    {
        int nready = Poll(clients, maxi+1, -1);

        if (clients[0].revents & POLLIN)
        {
            struct sockaddr_in clientsAddr;
            socklen_t len = sizeof(clientsAddr);
            int clientsFd = Accept(listenFd, (SA*)&clientsAddr, &len);
            printf("Connection from %s:%d\n",
                inet_ntop(AF_INET, &clientsAddr.sin_addr, buff, MAXLINE),
                ntohs(clientsAddr.sin_port));

            int isFull = 1;
            for (int i = 1; i < maxOpenFileCount; ++i)
            {
                if (clients[i].fd == -1)
                {
                    clients[i].fd = clientsFd;
                    clients[i].events = POLLIN;
                    maxi = max(maxi, i);
                    isFull = 0;
                    break;
                }
            }

            if(isFull)
                err_quit("server is busy now");

            if(--nready == 0)
                continue;
        }

        for (int i = 1; i <= maxi; ++i)
        {
            int sockFd = clients[i].fd;
            if(sockFd == -1) continue;
            if(clients[i].revents & (POLLIN | POLLERR))
            {
                int n = read(sockFd, buff, MAXLINE);
                if(n < 0)
                {
                    if(errno == ECONNRESET)
                    {
                        printf("clients[%d] aborted connection\n", i);
                        Close(sockFd);
                        clients[i].fd = -1;
                    }
                    else
                    {
                        err_sys("read error");
                    }
                }
                else if(n == 0)
                {
                    printf("clients[%d] closed connection\n", i);
                    Close(sockFd);
                    clients[i].fd = -1;
                }
                else
                {
                    buff[n] = 0;
                    Fputs(buff, stdout);
                    Writen(sockFd, buff, n);
                }

                if(--nready == 0)
                    break;
            }
        }
    }

    free(clients);
}
