#include "../apue/apue.h"

#include <sys/socket.h>
#include <arpa/inet.h>

#define SERVER_PORT 9877
#define LISTEN_QUEUE_LEN 256

typedef struct sockaddr SA;

typedef void (*sig_func_t)(int);
sig_func_t signal_local(int signo, sig_func_t func);

void sig_child(int signo);

void str_echo(int sockFd);
void str_cli(FILE* fp, int sockFd);

// socket api wrappers in sock_wrapper.c

int Socket(int family, int type, int protocol);
void Bind(int fd, const struct sockaddr *sa, socklen_t salen);
void Listen(int fd, int backlog);
void Connect(int fd, const struct sockaddr *sa, socklen_t salen);
void Setsockopt(int fd, int level, int optname, const void *optval, socklen_t optlen);

int Select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
       struct timeval *timeout);
