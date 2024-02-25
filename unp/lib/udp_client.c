#include    "unp.h"

int Udp_client(const char *host, const char *serv, SA **saptr, socklen_t *lenp)
{
    struct addrinfo hints;
    bzero(&hints, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;

    struct addrinfo *res, *ressave;

    int n = getaddrinfo(host, serv, &hints, &res);
    if (n != 0)
    {
        err_quit("udp_client error for %s, %s: %s",
                 host, serv, gai_strerror(n));
    }
    ressave = res;

    int sockfd = 0;
    do {
        sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (sockfd >= 0)
            break;
    } while ((res = res->ai_next) != NULL);

    if (res == NULL)
    {
        /* errno set from final socket() */
        err_sys("Udp_client error for %s, %s", host, serv);
    }

    *saptr = Malloc(res->ai_addrlen);
    memcpy(*saptr, res->ai_addr, res->ai_addrlen);
    *lenp = res->ai_addrlen;

    freeaddrinfo(ressave);

    return sockfd;
}
