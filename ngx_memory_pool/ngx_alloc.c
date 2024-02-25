/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */

#include "ngx_alloc.h"

ngx_uint_t  ngx_pagesize;
ngx_uint_t  ngx_pagesize_shift;
ngx_uint_t  ngx_cacheline_size;

void ngx_init_mem_params() {
    ngx_pagesize = getpagesize();
    // ngx_cacheline_size = NGX_CPU_CACHE_LINE;
    ngx_cacheline_size = 32;
    for (int n = ngx_pagesize; n >>= 1; ngx_pagesize_shift++);
}

void* ngx_alloc(size_t size) {
    void* p = malloc(size);
    if (p == NULL) {
        err_msg("malloc(%zu) failed", size);
    } else {
        //err_ret("malloc: %p:%zu", p, size);
    }
    return p;
}

void* ngx_calloc(size_t size)
{
    void* p = ngx_alloc(size);
    if (p) {
        ngx_memzero(p, size);
    }
    return p;
}

void* ngx_memalign(size_t alignment, size_t size) {
    void* p;
    int err = posix_memalign(&p, alignment, size);
    if (err != 0) {
        err_msg("posix_memalign(%zu, %zu) failed", alignment, size);
        p = NULL;
    } else {
        //err_ret("posix_memalign: %p:%zu @%zu", p, size, alignment);
    }
    return p;
}
