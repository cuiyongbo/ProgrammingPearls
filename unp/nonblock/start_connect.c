#include    "web.h"

void start_connect(struct File* fptr)
{
    struct addrinfo* ai = Host_serv(fptr->f_host, SERV, 0, SOCK_STREAM);
    int fd = Socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    fptr->f_fd = fd;
    printf("start_connect for %s, fd: %d\n", fptr->f_name, fd);

    int flags = Fcntl(fd, F_GETFL, 0);
    Fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    int n = connect(fd, ai->ai_addr, ai->ai_addrlen);
    if(n < 0)
    {
        if(errno != EINPROGRESS)
            err_sys("nonblocking connect error");

        fptr->f_flags = F_CONNECTING;
        FD_SET(fd, &g_rset);
        FD_SET(fd, &g_wset);
        g_maxfd = max(g_maxfd, fd);
    }
    else
    {
        write_get_cmd(fptr);
    }
}
