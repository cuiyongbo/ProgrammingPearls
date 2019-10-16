#include    "icmpd.h"

int readable_listen(void)
{
    socklen_t clilen = sizeof(cliaddr);
    int connfd = Accept(g_listenfd, (SA*)&cliaddr, &clilen);

    int i = 0;
    for (; i < FD_SETSIZE; i++)
    {
        if (g_client[i].connfd < 0)
        {
            g_client[i].connfd = connfd;
            break;
        }
    }

    if (i == FD_SETSIZE)
    {
        close(connfd);      /* can't handle new client, */
        return --g_nready;   /* rudely close the new connection */
    }

    printf("new connection, i = %d, connfd = %d\n", i, connfd);

    FD_SET(connfd, &g_allset);    /* add new descriptor to set */
    g_maxi = max(i, g_maxi);
    g_maxfd = max(g_maxfd, connfd);
    return --g_nready;
}
