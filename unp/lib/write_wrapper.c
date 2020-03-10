#include "unp.h"

ssize_t write_wrapper(int fd, const void* vptr, size_t n)
{
    const char* ptr = (const char*)vptr;
    size_t nleft = n;
    while(nleft > 0)
    {
        ssize_t nwritten = write(fd, ptr, nleft);
        if(nwritten <= 0)
        {
            if(nwritten < 0 && errno == EINTR)
                nwritten = 0;
            else
                return -1;
        }

        nleft -= nwritten;
        ptr += nwritten;
    }
    return n;
}

void Writen(int fd, const void* ptr, size_t nbytes)
{
    if(write_wrapper(fd, ptr, nbytes) != nbytes)
        err_sys("Write error");
}
