#pragma once

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wint-to-void-pointer-cast"
#  pragma clang diagnostic ignored "-Wgnu-folding-constant"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
#  pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
#  pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <syslog.h>
#include <pthread.h>

#include <poll.h>
#include <sys/select.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <netdb.h>

#define MAXLINE 4096
#define BUFFSIZE    8192    /* buffer size for reads and writes */

#define SERVER_PORT 9877
#define LISTEN_QUEUE_LEN 256

#define UNIXSTR_PATH    "/tmp/unix.str" /* Unix domain stream */
#define UNIXDG_PATH     "/tmp/unix.dg"  /* Unix domain datagram */

/* default file access permissions for new files and new directories */
#define FILE_MODE   (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
#define DIR_MODE    (FILE_MODE | S_IXUSR | S_IXGRP | S_IXOTH)

#define max(a, b) ((a)>(b)?(a):(b))
#define min(a, b) ((a)<(b)?(a):(b))
#define element_of(arr) (sizeof(arr)/sizeof(arr[0]))

typedef struct sockaddr SA;

#if !defined(SCM_CREDENTIALS)
#define SCM_CREDENTIALS 0x02
#define CMGROUP_MAX 16
struct cmsgcred
{
    pid_t   cmcred_pid;             /* PID of sending process */
    uid_t   cmcred_uid;             /* real UID of sending process */
    uid_t   cmcred_euid;            /* effective UID of sending process */
    gid_t   cmcred_gid;             /* real GID of sending process */
    short   cmcred_ngroups;         /* number or groups */
    gid_t   cmcred_groups[CMGROUP_MAX];     /* groups */
};
#endif

// log utilities
void err_sys(const char *fmt, ...);
void err_ret(const char *fmt, ...);
void err_dump(const char *fmt, ...);
void err_msg(const char *fmt, ...);
void err_quit(const char *fmt, ...);

// daemonization utility in daemon_helpers.c
int daemon_init(const char* pname, int facility);
void daemon_inetd(const char *pname, int facility);

// stdio api wrappers in stdio_wrapper.c
FILE* Fopen(const char *filename, const char *mode);
void Fclose(FILE *fp);
FILE* Fdopen(int fd, const char *type);
char * Fgets(char *ptr, int n, FILE *stream);
void Fputs(const char *ptr, FILE *stream);

// signal in signal.c
typedef void (*sig_func_t)(int);
sig_func_t Signal(int signo, sig_func_t func);

void str_echo(int sockFd);
void str_cli(FILE* fp, int sockFd);
void dg_cli(FILE* fp, int sockfd, const SA* pservaddr, socklen_t servlen);
void dg_echo(int sockfd, SA* pcliaddr, socklen_t clilen);

// signal mask related api wrappers in unix_api_wrapper.c
void Sigaddset(sigset_t *set, int signo);
void Sigdelset(sigset_t *set, int signo);
void Sigemptyset(sigset_t *set);
void Sigfillset(sigset_t *set);
int Sigismember(const sigset_t *set, int signo);
void Sigpending(sigset_t *set);
void Sigprocmask(int how, const sigset_t *set, sigset_t *oset);

// socket api wrappers in sock_wrapper.c
int Socket(int family, int type, int protocol);
void Bind(int fd, const struct sockaddr *sa, socklen_t salen);
void Listen(int fd, int backlog);
int Accept(int fd, struct sockaddr *sa, socklen_t *salenptr);
void Connect(int fd, const struct sockaddr *sa, socklen_t salen);

void Getsockopt(int fd, int level, int optname, void *optval, socklen_t *optlenptr);
void Setsockopt(int fd, int level, int optname, const void *optval, socklen_t optlen);
char* Sock_ntop_host(const struct sockaddr *sa, socklen_t salen);
void Getpeername(int fd, struct sockaddr *sa, socklen_t *salenptr);
void Getsockname(int fd, struct sockaddr *sa, socklen_t *salenptr);

void Inet_pton(int family, const char *strptr, void *addrptr);
const char* Inet_ntop(int family, const void *addrptr, char *strptr, size_t len);

int Select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
int Poll(struct pollfd *fdarray, unsigned long nfds, int timeout);
void Shutdown(int fd, int how);
void Socketpair(int family, int type, int protocol, int *fd);

ssize_t Readn(int fd, void *ptr, size_t nbytes);
ssize_t Readline(int fd, void* ptr, size_t maxlen);
void Writen(int fd, const void* ptr, size_t nbytes);

ssize_t Recv(int fd, void *ptr, size_t nbytes, int flags);
ssize_t Recvfrom(int fd, void *ptr, size_t nbytes, int flags,
         struct sockaddr *sa, socklen_t *salenptr);
ssize_t Recvmsg(int fd, struct msghdr *msg, int flags);

void Send(int fd, const void *ptr, size_t nbytes, int flags);
void Sendto(int fd, const void *ptr, size_t nbytes, int flags,
       const struct sockaddr *sa, socklen_t salen);
void Sendmsg(int fd, const struct msghdr *msg, int flags);

// in unix_api_wrapper.c
pid_t Fork();
pid_t Wait(int *iptr);
pid_t Waitpid(pid_t pid, int *iptr, int options);
void* Malloc(size_t size);
void* Calloc(size_t n, size_t size);
ssize_t Read(int fd, void *ptr, size_t nbytes);
void Write(int fd, const void* ptr, size_t nbytes);
void Close(int fd);

int Fcntl(int fd, int cmd, int arg);
int Ioctl(int fd, int request, void *arg);

// in readable_timeo.c
int readable_timeo(int fd, int sec);

// in tcp_helpers.c
int Tcp_connect(const char *host, const char *serv);
int Tcp_listen(const char *host, const char *serv, socklen_t *addrlenp);

// int gf_time.c, output time in format such as 00:00:00.000000
char* gf_time(void);

// a non-blocked connect in connect_nonb.c
int connect_nonb(int sockfd, const SA *saptr, socklen_t salen, int nsec);

// in host_serv.c
struct addrinfo* Host_serv(const char *host, const char *serv, int family, int socktype);

/* The structure returned by recvfrom_flags() */
struct unp_in_pktinfo
{
    struct in_addr    ipi_addr;   /* dst IPv4 address */
    int               ipi_ifindex;/* received interface index */
};

int Open(const char *pathname, int oflag, mode_t mode);
void Unlink(const char *pathname);
int Mkstemp(char *template);
void* Mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset);
void Dup2(int fd1, int fd2);

// in read_fd.c
ssize_t Read_fd(int fd, void *ptr, size_t nbytes, int *recvfd);
ssize_t Write_fd(int fd, void *ptr, size_t nbytes, int sendfd);

int Sock_get_port(const struct sockaddr *sa, socklen_t salen);
int Sock_bind_wild(int sockfd, int family);
