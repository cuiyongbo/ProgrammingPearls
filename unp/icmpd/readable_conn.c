#include "icmpd.h"

int readable_conn(int i)
{
    int sock_bind_wild(int sockfd, int family);
    int sock_get_port(const struct sockaddr *sa, socklen_t salen);

    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);

    char c;
    int recvfd = -1;
    int unixfd = g_client[i].connfd;
    if(Read_fd(unixfd, &c, 1, &recvfd) == 0)
    {
        err_msg("client %d terminated, recvfd=%d", i, recvfd);
        goto client_done;
    }

    if (recvfd < 0)
    {
        err_msg("Read_fd did not return descriptor");
        goto client_err;
    }

    if(getsockname(recvfd, (SA*)&ss, &len) < 0)
    {
        err_ret("getsockname error");
        goto client_err;
    }

    g_client[i].family = ss.ss_family;
    g_client[i].lport = sock_get_port((SA*)&ss, len);
    if (g_client[i].lport == 0)
    {
        g_client[i].lport = sock_bind_wild(recvfd, g_client[i].family);
        if(g_client[i].lport <= 0)
        {
            err_ret("error binding ephemeral port");
            goto client_err;
        }
    }

    Write(unixfd, "1", 1);
    Close(recvfd);
    return --g_nready;

client_err:
    Write(unixfd, "0", 1);
client_done:
    Close(unixfd);
    if (recvfd > 0)
    {
        Close(recvfd);
    }
    FD_CLEAR(unixfd, &g_allset);
    g_client[i].connfd = -1;
    return --g_nready;
}
