#include    "icmpd.h"

int main(int argc, char **argv)
{
    g_maxi = -1;                  /* index into client[] array */
    for (int i = 0; i < FD_SETSIZE; i++)
        g_client[i].connfd = -1;  /* -1 indicates available entry */
    FD_ZERO(&g_allset);

    g_fd4 = Socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    FD_SET(g_fd4, &g_allset);
    g_maxfd = g_fd4;

#ifdef  IPV6
    g_fd6 = Socket(AF_INET6, SOCK_RAW, IPPROTO_ICMPV6);
    FD_SET(g_fd6, &g_allset);
    g_maxfd = max(g_maxfd, g_fd6);
#endif

    g_listenfd = Socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un sun;
    bzero(&sun, sizeof(sun));
    sun.sun_family = AF_LOCAL;
    strcpy(sun.sun_path, ICMPD_PATH);
    unlink(ICMPD_PATH);
    Bind(g_listenfd, (SA *)&sun, sizeof(sun));

    Listen(g_listenfd, LISTENQ);

    FD_SET(g_listenfd, &g_allset);
    g_maxfd = max(g_maxfd, g_listenfd);

    for( ; ; )
    {
        g_rset = g_allset;
        g_nready = Select(g_maxfd+1, &g_rset, NULL, NULL, NULL);

        if (FD_ISSET(g_listenfd, &g_rset))
            if (readable_listen() <= 0)
                continue;

        if (FD_ISSET(fd4, &g_rset))
            if (readable_v4() <= 0)
                continue;

#ifdef  IPV6
        if (FD_ISSET(fd6, &g_rset))
            if (readable_v6() <= 0)
                continue;
#endif

        for (int i = 0; i <= g_maxi; i++)
        {
            if (g_client[i].connfd < 0)
                continue;

            if (FD_ISSET(g_client[i].connfd, &g_rset))
            {
                if (readable_conn(i) <= 0)
                    break;              /* no more readable descriptors */
            }
        }
    }
    exit(0);
}
