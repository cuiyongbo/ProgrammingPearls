#include    "unp.h"

struct addrinfo* Host_serv(const char *host, const char *serv, int family, int socktype)
{
    struct addrinfo hints;
    bzero(&hints, sizeof(struct addrinfo));
    hints.ai_flags = AI_CANONNAME;  /* always return canonical name */
    hints.ai_family = family;       /* 0, AF_INET, AF_INET6, etc. */
    hints.ai_socktype = socktype;   /* 0, SOCK_STREAM, SOCK_DGRAM, etc. */

    struct addrinfo* res;
    int n = getaddrinfo(host, serv, &hints, &res);
    if (n != 0)
    {
        err_quit("host_serv error for %s, %s: %s",
                 (host == NULL) ? "(no hostname)" : host,
                 (serv == NULL) ? "(no service name)" : serv,
                 gai_strerror(n));
    }

    return res;  /* return pointer to first on linked list */
}
