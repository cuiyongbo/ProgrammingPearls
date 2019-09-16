#include "../apue/apue.h"

#include <sys/socket.h>
#include <arpa/inet.h>

#define SERVER_PORT 9877
#define LISTEN_QUEUE_LEN 256

typedef struct sockaddr SA;

void str_echo(int sockFd);
void str_cli(FILE* fp, int sockFd);

