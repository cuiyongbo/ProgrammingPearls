#include    "unp.h"

static ssize_t readn(int fd, void *vptr, size_t n)
{
    char* ptr = vptr;
    ssize_t nleft = n;
    while (nleft > 0)
    {
        ssize_t nread = read(fd, ptr, nleft);
        if (nread < 0)
        {
            if (errno == EINTR)
                nread = 0;      /* and call read() again */
            else
                return -1;
        }
        else if (nread == 0)
            break;              /* EOF */

        nleft -= nread;
        ptr   += nread;
    }
    return (n - nleft);      /* return >= 0 */
}

ssize_t Readn(int fd, void *ptr, size_t nbytes)
{
    ssize_t     n;
    if ((n = readn(fd, ptr, nbytes)) < 0)
        err_sys("readn error");
    return n;
}
