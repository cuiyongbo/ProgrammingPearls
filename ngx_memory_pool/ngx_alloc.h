/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */

#ifndef __NGX_ALLOC_H_INCLUDE_
#define __NGX_ALLOC_H_INCLUDE_

#include "ngx_core.h"

void *ngx_alloc(size_t size);
void *ngx_calloc(size_t size);

#define ngx_free          free
#define ngx_memzero(buf, n)       (void) memset(buf, 0, n)
#define ngx_memset(buf, c, n)     (void) memset(buf, c, n)

void *ngx_memalign(size_t alignment, size_t size);

extern ngx_uint_t  ngx_pagesize;
extern ngx_uint_t  ngx_pagesize_shift;
extern ngx_uint_t  ngx_cacheline_size;

// called *BEFORE* calling any ngx_mem* functions
void ngx_init_mem_params();

#endif