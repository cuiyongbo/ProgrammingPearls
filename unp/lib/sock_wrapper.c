#include "unp.h"

void Getpeername(int fd, struct sockaddr *sa, socklen_t *salenptr)
{
    if (getpeername(fd, sa, salenptr) < 0)
        err_sys("getpeername error");
}

void Getsockname(int fd, struct sockaddr *sa, socklen_t *salenptr)
{
    if (getsockname(fd, sa, salenptr) < 0)
        err_sys("getsockname error");
}

void Getsockopt(int fd, int level, int optname, void *optval, socklen_t *optlenptr)
{
    if (getsockopt(fd, level, optname, optval, optlenptr) < 0)
        err_sys("getsockopt error");
}

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

int Poll(struct pollfd *fdarray, unsigned long nfds, int timeout)
{
    int n = poll(fdarray, nfds, timeout);
    if(n < 0)
        err_sys("poll error");
    return n;
}

void Shutdown(int fd, int how)
{
    if(shutdown(fd, how) < 0)
        err_sys("shutdown error");
}

void Socketpair(int family, int type, int protocol, int *fd)
{
    int  n;
    if ((n = socketpair(family, type, protocol, fd)) < 0)
        err_sys("socketpair error");
}

ssize_t Recv(int fd, void *ptr, size_t nbytes, int flags)
{
    ssize_t     n;
    if ((n = recv(fd, ptr, nbytes, flags)) < 0)
        err_sys("recv error");
    return n;
}

ssize_t Recvfrom(int fd, void *ptr, size_t nbytes, int flags,
         struct sockaddr *sa, socklen_t *salenptr)
{
    ssize_t     n;
    if ((n = recvfrom(fd, ptr, nbytes, flags, sa, salenptr)) < 0)
        err_sys("recvfrom error");
    return n;
}

ssize_t Recvmsg(int fd, struct msghdr *msg, int flags)
{
    ssize_t     n;
    if ((n = recvmsg(fd, msg, flags)) < 0)
        err_sys("recvmsg error");
    return n;
}

void Send(int fd, const void *ptr, size_t nbytes, int flags)
{
    if (send(fd, ptr, nbytes, flags) != (ssize_t)nbytes)
        err_sys("send error");
}

void Sendto(int fd, const void *ptr, size_t nbytes, int flags,
       const struct sockaddr *sa, socklen_t salen)
{
    if (sendto(fd, ptr, nbytes, flags, sa, salen) != (ssize_t)nbytes)
        err_sys("sendto error");
}

void Sendmsg(int fd, const struct msghdr *msg, int flags)
{
    ssize_t nbytes = 0;
    for (unsigned int i = 0; i < msg->msg_iovlen; i++)
        nbytes += msg->msg_iov[i].iov_len;

    if (sendmsg(fd, msg, flags) != nbytes)
        err_sys("sendmsg error");
}

const char* Inet_ntop(int family, const void *addrptr, char *strptr, size_t len)
{
    if(strptr == NULL)
        return NULL;

    const char* ptr = inet_ntop(family, addrptr, strptr, len);
    if(ptr == NULL)
        err_sys("inet_ntop error");
    return ptr;
}

void Inet_pton(int family, const char* strptr, void* addrptr)
{
    int n = inet_pton(family, strptr, addrptr);
    if(n < 0)
        err_sys("inet_pton error for %s", strptr);
    else if(n == 0)
        err_quit("inet_pton error for %s", strptr);
}
