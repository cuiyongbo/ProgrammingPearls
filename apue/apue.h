#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MAXLINE 4096
#define FILE_MODE (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
#define DIR_MODE (FILE_MODE | S_IXUSR | S_IXGRP | S_IXOTH)

#define err_sys(...) {	\
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, ": %s\n", strerror(errno)); \
	exit(EXIT_FAILURE); }

#define err_quit(...) {	\
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, "\n");\
	exit(EXIT_FAILURE); }

#define err_msg(...) {	\
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, ": %s\n", strerror(errno)); \
	}


#define err_ret(...) {	\
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, "\n");\
	}

#define err_dump(...) {	\
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, ": %s\n", strerror(errno)); \
	abort();		\
	exit(EXIT_FAILURE); }

typedef ssize_t (*UnixDomainSocketUserFunc) (int, const void*, size_t);
int send_fd(int fd, int fd_to_send); /* fig 17.13 */
int send_err(int fd, int errcode, const char* errmsg); /* fig 17.12 */
int recv_fd(int fd, UnixDomainSocketUserFunc func); /* fig 17.14 */

