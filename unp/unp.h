#include "../apue/apue.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <poll.h>

#define SERVER_PORT 9877
#define LISTEN_QUEUE_LEN 256

#if !defined(OPEN_MAX)
#define OPEN_MAX FOPEN_MAX
#endif

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

int Select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
int Poll(struct pollfd *fdarray, unsigned long nfds, int timeout);

ssize_t Readline(int fd, void* ptr, size_t maxlen);
void Writen(int fd, const void* ptr, size_t nbytes);

void Shutdown(int fd, int how);

char* Sock_ntop_host(const struct sockaddr *sa, socklen_t salen);

// in unix_api_wrapper.c
void* Malloc(size_t size);
ssize_t Read(int fd, void *ptr, size_t nbytes);
void Write(int fd, const void* ptr, size_t nbytes);
void Close(int fd);
