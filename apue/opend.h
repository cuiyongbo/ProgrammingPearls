#include "apue.h"

#define CL_OPEN "open"
#define MAXARGC	50
#define WHITESPACE " "

extern char errmsg[BUFSIZ];
extern int oflag;
extern char* pathname;

int buf_args(char*, int (*openfunc)(int, char**));
int cli_args(int, char**);
void handle_request(char*, int, int);

