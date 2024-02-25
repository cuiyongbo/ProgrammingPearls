#include "unp.h"

void* Malloc(size_t size)
{
    void* ptr = malloc(size);
    if (ptr == NULL)
        err_sys("malloc error");
    return ptr;
}

void* Calloc(size_t n, size_t size)
{
    void* ptr = calloc(n, size);
    if(ptr == NULL)
        err_sys("calloc error");
    return ptr;
}

pid_t Fork()
{
    pid_t pid = fork();
    if(pid == -1)
        err_sys("fork error");
    return pid;
}

pid_t Wait(int *iptr)
{
    pid_t pid;
    if ((pid = wait(iptr)) == -1)
        err_sys("wait error");
    return pid;
}

pid_t Waitpid(pid_t pid, int *iptr, int options)
{
    pid_t retpid;
    if ((retpid = waitpid(pid, iptr, options)) == -1)
        err_sys("waitpid error");
    return retpid;
}

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

void Close(int fd)
{
    if (close(fd) == -1)
        err_sys("close error");
}

void Sigaddset(sigset_t *set, int signo)
{
    if (sigaddset(set, signo) == -1)
        err_sys("sigaddset error");
}

void Sigdelset(sigset_t *set, int signo)
{
    if (sigdelset(set, signo) == -1)
        err_sys("sigdelset error");
}

void Sigemptyset(sigset_t *set)
{
    if (sigemptyset(set) == -1)
        err_sys("sigemptyset error");
}

void Sigfillset(sigset_t *set)
{
    if (sigfillset(set) == -1)
        err_sys("sigfillset error");
}

int Sigismember(const sigset_t *set, int signo)
{
    int n = sigismember(set, signo);
    if (n == -1)
        err_sys("sigismember error");
    return n;
}

void Sigpending(sigset_t *set)
{
    if (sigpending(set) == -1)
        err_sys("sigpending error");
}

void Sigprocmask(int how, const sigset_t *set, sigset_t *oset)
{
    if (sigprocmask(how, set, oset) == -1)
        err_sys("sigprocmask error");
}

int Fcntl(int fd, int cmd, int arg)
{
    int n;
    if((n=fcntl(fd, cmd, arg)) == -1)
        err_sys("fcntl error");
    return n;
}

int Ioctl(int fd, int request, void *arg)
{
    int n = ioctl(fd, request, arg);
    if (n == -1)
        err_sys("ioctl error");
    return n;
}

int Mkstemp(char *template)
{
    int i;

#ifdef HAVE_MKSTEMP
    if ((i = mkstemp(template)) < 0)
        err_quit("mkstemp error");
#else
    if (mktemp(template) == NULL || template[0] == 0)
        err_quit("mktemp error");
    i = Open(template, O_CREAT | O_WRONLY, FILE_MODE);
#endif

    return i;
}

int Open(const char *pathname, int oflag, mode_t mode)
{
    int fd;
    if ((fd = open(pathname, oflag, mode)) == -1)
        err_sys("open error for %s", pathname);
    return fd;
}

void Unlink(const char *pathname)
{
    if (unlink(pathname) == -1)
        err_sys("unlink error for %s", pathname);
}

void* Mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset)
{
    void    *ptr;
    if ((ptr = mmap(addr, len, prot, flags, fd, offset)) == MAP_FAILED)
        err_sys("mmap error");
    return ptr;
}

void Dup2(int fd1, int fd2)
{
    if (dup2(fd1, fd2) == -1)
        err_sys("dup2 error");
}
