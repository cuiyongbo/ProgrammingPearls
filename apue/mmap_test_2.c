#include "apue.h"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>  


int main()
{
    char* name = "test-hello";
    int fd = shm_open(name, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR|S_IXUSR);
    if(fd < 0)
    {
        err_sys("shm_open");
    }

    int size = 1024;
    if(ftruncate(fd, size)<0)
    {
        err_sys("ftrucate error");
    }

    void* addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if(addr == MAP_FAILED)
    {
        err_sys("mmap failed");
    }

    char* buffer = "hello world";
    int len = strlen(buffer);
    memcpy(addr, buffer, len);

    sleep(1000);

    if(munmap(addr, size)<0)
    {
        err_sys("munmap error");
    }

    if(shm_unlink(name) < 0)
    {
        err_sys("shm_unlink error");
    }
}
