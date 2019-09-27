#include "unp.h"

void Setsockopt(int fd, int level, int optname, const void *optval, socklen_t optlen)
{
    if (setsockopt(fd, level, optname, optval, optlen) < 0)
        err_sys("setsockopt error");
}

int Socket(int family, int type, int protocol)
{
    int fd = socket(family, type, protocol);
    if ( fd < 0)
        err_sys("socket error");
    return fd;
}

void Bind(int fd, const struct sockaddr *sa, socklen_t salen)
{
    if (bind(fd, sa, salen) < 0)
        err_sys("bind error");
}

void Listen(int fd, int backlog)
{
    if (listen(fd, backlog) < 0)
        err_sys("listen error");
}

int Accept(int fd, struct sockaddr *sa, socklen_t *salenptr)
{
    int  n;

again:
    if ((n = accept(fd, sa, salenptr)) < 0)
    {
#ifdef  EPROTO
        if (errno == EPROTO || errno == ECONNABORTED)
#else
        if (errno == ECONNABORTED)
#endif
            goto again;
        else
            err_sys("accept error");
    }
    return n;
}

void Connect(int fd, const struct sockaddr *sa, socklen_t salen)
{
    if (connect(fd, sa, salen) < 0)
        err_sys("connect error");
}

int Select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
       struct timeval *timeout)
{
    int n = select(nfds, readfds, writefds, exceptfds, timeout);
    if(n < 0)
        err_sys("select error");
    return n;
}

void Shutdown(int fd, int how)
{
    if(shutdown(fd, how) < 0)
        err_sys("shutdown error");
}
