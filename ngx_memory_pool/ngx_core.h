/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */

#ifndef _NGX_CORE_H_INCLUDE_
#define _NGX_CORE_H_INCLUDE_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>

typedef int ngx_int_t;
typedef unsigned int ngx_uint_t;
typedef int                      ngx_fd_t;


typedef struct ngx_pool_s            ngx_pool_t;
typedef struct ngx_chain_s           ngx_chain_t;

#define  NGX_OK          0
#define  NGX_ERROR      -1
#define  NGX_AGAIN      -2
#define  NGX_BUSY       -3
#define  NGX_DONE       -4
#define  NGX_DECLINED   -5
#define  NGX_ABORT      -6


#ifndef ngx_inline
#define ngx_inline      inline
#endif

#ifndef NGX_ALIGNMENT
#define NGX_ALIGNMENT   sizeof(unsigned long)    /* platform word */
#endif

#define ngx_align(d, a)     (((d) + (a - 1)) & ~(a - 1))
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) a - 1)) & ~((uintptr_t) a - 1))


#define ngx_abort       abort


#define NGX_INVALID_FILE         -1
#define NGX_FILE_ERROR           -1

#define ngx_close_file           close
#define ngx_close_file_n         "close()"

#define ngx_delete_file(name)    unlink((const char *) name)
#define ngx_delete_file_n        "unlink()"

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


#include "ngx_errno.h"
#include "ngx_alloc.h"
#include "ngx_palloc.h"

#endif