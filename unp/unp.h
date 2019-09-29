#include "../apue/apue.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <poll.h>
#include <netdb.h>

#define SERVER_PORT 9877
#define LISTEN_QUEUE_LEN 256

typedef struct sockaddr SA;

// stdio api wrappers in stdio_wrapper.c
FILE* Fopen(const char *filename, const char *mode);
void Fclose(FILE *fp);
FILE* Fdopen(int fd, const char *type);
char * Fgets(char *ptr, int n, FILE *stream);
void Fputs(const char *ptr, FILE *stream);

// signal in signal.c
typedef void (*sig_func_t)(int);
sig_func_t Signal(int signo, sig_func_t func);

void sig_child(int signo);

void str_echo(int sockFd);
void str_cli(FILE* fp, int sockFd);

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

void Setsockopt(int fd, int level, int optname, const void *optval, socklen_t optlen);
char* Sock_ntop_host(const struct sockaddr *sa, socklen_t salen);

int Select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
int Poll(struct pollfd *fdarray, unsigned long nfds, int timeout);
void Shutdown(int fd, int how);

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
void* Malloc(size_t size);
ssize_t Read(int fd, void *ptr, size_t nbytes);
void Write(int fd, const void* ptr, size_t nbytes);
void Close(int fd);

int Fcntl(int fd, int cmd, int arg);

// in readable_timeo.c
int readable_timeo(int fd, int sec);

// in tcp_helpers.c
int Tcp_connect(const char *host, const char *serv);
int Tcp_listen(const char *host, const char *serv, socklen_t *addrlenp);
