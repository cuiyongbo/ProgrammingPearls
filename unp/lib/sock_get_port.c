#include    "unp.h"

int sock_get_port(const struct sockaddr *sa, socklen_t salen)
{
    int port = -1;
    switch (sa->sa_family)
    {
    case AF_INET:
    {
        struct sockaddr_in  *sin = (struct sockaddr_in *) sa;
        port = sin->sin_port;
    }

#ifdef  IPV6
    case AF_INET6:
    {
        struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *) sa;
        port = sin6->sin6_port;
    }
#endif
    }
    return port;
}

int Sock_get_port(const struct sockaddr *sa, socklen_t salen)
{
    int port = sock_get_port(sa, salen);
    if(port < 0)
    {
        err_sys("address family not supported");
    }
    return port;
}
