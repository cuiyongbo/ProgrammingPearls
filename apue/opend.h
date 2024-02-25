#include "apue.h"

#define CS_OPEN "/tmp/opend.socket"
#define CL_OPEN "open"
#define MAXARGC	50
#define WHITESPACE " "

extern int debug; /* nonzero if interactive (not daemon) */
extern char errmsg[BUFSIZ];
extern int oflag;
extern char* pathname;

typedef struct
{
	int fd;
	uid_t uid;
} ClientInfo;

extern ClientInfo* clientInfos;
extern int clientCount;

int client_add(int, uid_t);
void client_del(int);
void loop();
void handle_request_02(char*, int, int, uid_t);

int buf_args(char*, int (*openfunc)(int, char**));
int cli_args(int, char**);
void handle_request(char*, int, int);

