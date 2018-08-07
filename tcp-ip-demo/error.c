#include <errno.h>
#include <stdarg.h>
#include "cli_serv.h"

#define MAXLINE	4096

static void err_doit(int, const char*, va_list);

/* Fatal error related to a system call.
 * Print a message and return. */

void err_ret(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	err_doit(1, fmt, ap);
	va_end(ap);
	return;
}

/* Fatal error related to a system call.
 * Print a message and terminate. */

void err_sys(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	err_doit(1, fmt, ap);
	va_end(ap);
	exit(EXIT_FAILURE);
}

/* Fatal error related to a system call.
 * Print a message, dump core, and terminate. */

void err_dump(const char *fmt, ...)
{
	va_list		ap;

	va_start(ap, fmt);
	err_doit(1, fmt, ap);
	va_end(ap);
	abort();		/* dump core and terminate */
	exit(EXIT_FAILURE);		/* shouldn't get here */
}

/* Fatal error unrelated to a system call.
 * Print a message and terminate. */

void err_quit(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	err_doit(0, fmt, ap);
	va_end(ap);
	exit(EXIT_FAILURE);
}

/* Fatal error unrelated to a system call.
 * Print a message and return. */

void err_msg(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	err_doit(0, fmt, ap);
	va_end(ap);
	return;
}

/* Print a message and return to caller.
 * Caller specifies "requestErrMsg". */

void err_doit(int requestErrMsg, const char* fmt, va_list ap)
{
	int duplicateErrno = errno;

	char buf[MAXLINE];
	vsprintf(buf, fmt, ap);
	if (requestErrMsg)
		sprintf(buf+strlen(buf), ": %s", strerror(duplicateErrno));
	strcat(buf, "\n");
	fflush(stdout); /*in case stdout and stderr are the same*/
	fputs(buf, stderr);
	fflush(stderr);
	return;
}


