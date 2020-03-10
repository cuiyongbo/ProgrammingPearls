#include "unp_thread.h"

static pthread_mutex_t* g_mptr;  /* actual mutex will be in shared memory */

void my_lock_init(char *pathname)
{
    int fd = Open("/dev/zero", O_RDWR, 0);
    g_mptr = Mmap(0, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, 0);
    Close(fd);

    pthread_mutexattr_t mattr;
    Pthread_mutexattr_init(&mattr);
    Pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    Pthread_mutex_init(g_mptr, &mattr);
}

void my_lock_wait()
{
    Pthread_mutex_lock(g_mptr);
}

void my_lock_release()
{
    Pthread_mutex_unlock(g_mptr);
}
