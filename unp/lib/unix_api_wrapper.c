#include "unp.h"

ssize_t Read(int fd, void *ptr, size_t nbytes)
{
    ssize_t n = read(fd, ptr, nbytes);
    if ( n == -1)
        err_sys("read error");
    return n;
}

void Write(int fd, const void *ptr, size_t nbytes)
{
    if (write(fd, ptr, nbytes) != nbytes)
        err_sys("write error");
}
