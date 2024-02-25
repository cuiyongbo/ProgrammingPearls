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
#include <syslog.h>

#include <pthread.h>

#define MAXLINE 4096
#define FILE_MODE (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
#define DIR_MODE (FILE_MODE | S_IXUSR | S_IXGRP | S_IXOTH)

#define max(a, b) ((a)>(b)?(a):(b))
#define min(a, b) ((a)<(b)?(a):(b))
#define element_of(arr) (sizeof(arr)/sizeof(arr[0]))

#define err_sys(...) do {	\
	fprintf(stderr, "%s(%d): ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, ": %s\n", strerror(errno)); \
	exit(EXIT_FAILURE); } while(0)

#define err_quit(...) do {	\
	fprintf(stderr, "%s(%d): ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, "\n");\
	exit(EXIT_FAILURE); } while(0)

#define err_msg(...) do {	\
	fprintf(stderr, "%s(%d): ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, ": %s\n", strerror(errno)); \
	} while(0)

#define err_ret(...) do {	\
	fprintf(stderr, "%s(%d): ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, "\n");\
	} while(0)

#define err_dump(...) do {	\
	fprintf(stderr, "%s(%d): ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__);\
	fprintf(stderr, ": %s\n", strerror(errno)); \
	abort(); } while(0)

#define err_exit(err, ...) do {    \
    fprintf(stderr, __VA_ARGS__);   \
    fprintf(stderr, ": %s\n", strerror(err));   \
    exit(EXIT_FAILURE); } while(0)

#define err_cont(err, ...) do {    \
    fprintf(stderr, __VA_ARGS__);   \
    fprintf(stderr, ": %s\n", strerror(err));   \
    } while(0)

int send_fd(int fd, int fd_to_send); /* fig 17.13 */
int send_err(int fd, int errcode, const char* errmsg); /* fig 17.12 */

typedef ssize_t (*UnixDomainSocketUserFunc) (int, const void*, size_t);
int recv_fd(int fd, UnixDomainSocketUserFunc func); /* fig 17.14 */

int fd_pipe(int fd[2]); /* fig 17.2 */

int serv_accept(int listenfd, uid_t* uidptr);
int serv_listen(const char* name);
int cli_conn(const char* name);

/* fig 10.24 implementation using signal */
void TELL_WAIT(void);
void TELL_PARENT(pid_t pid);
void TELL_CHILD(pid_t pid);
void WAIT_PARENT(void);
void WAIT_CHILD(void);

int set_cloexec(int fd);	/* fig 13.9*/
void daemonize(const char* cmd); /* fig 13.1 */

typedef void* (thread_func_t)(void*);

int makeThread(thread_func_t func, void*);

void pr_exit(int status); /* fig 8.5 */
void pr_mask(const char *str);

int lockfile(int fd);
int already_running();
