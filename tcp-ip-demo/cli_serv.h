#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define REQUEST_SIZE	400
#define REPLY_SIZE	400

#define UDP_SERVER_PORT	7777
#define TCP_SERVER_PORT	8888
#define TTCP_SERVER_PORT	9999

#define UNUSED_VAR(o) ((void)o)
#define element_of_array(arr) (sizeof(arr)/sizeof((arr)[0]))

#define SA struct sockaddr*

void err_quit(const char*, ...);
void err_ret(const char*, ...);
void err_sys(const char*, ...);
void err_dump(const char*, ...);
void err_msg(const char*, ...);

int read_stream(int, char*, int);


