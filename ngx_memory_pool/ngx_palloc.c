/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */

#include "ngx_core.h"

static ngx_inline void* ngx_palloc_small(ngx_pool_t* pool, size_t size, ngx_uint_t align);
static void* ngx_palloc_block(ngx_pool_t* pool, size_t size);
static void* ngx_palloc_large(ngx_pool_t* pool, size_t size);

ngx_pool_t* ngx_create_pool(size_t size) {
    ngx_pool_t* p = ngx_memalign(NGX_POOL_ALIGNMENT, size);
    if (p == NULL) {
        return NULL;
    }

    p->current = p;
    p->d.last = (u_char*)p + sizeof(ngx_pool_t);
    p->d.end = (u_char*)p + size;
    p->d.next = NULL;
    p->d.failed = 0;

    size = size - sizeof(ngx_pool_t);
    p->max = (size < NGX_MAX_ALLOC_FROM_POOL) ? size : NGX_MAX_ALLOC_FROM_POOL;

    p->chain = NULL;
    p->large = NULL;
    p->cleanup = NULL;

    return p;
}

void ngx_destroy_pool(ngx_pool_t* pool)
{
    for (ngx_pool_cleanup_t* c = pool->cleanup; c; c = c->next)
    {
        if (c->handler)
        {
            err_ret("run cleanup: %p", c);
            c->handler(c->data);
        }
    }

    for (ngx_pool_large_t* l = pool->large; l; l = l->next)
    {
        if (l->alloc)
            ngx_free(l->alloc);
    }

    for (ngx_pool_t* p = pool;  p; )
    {
        ngx_pool_t* next = p->d.next;
        ngx_free(p);
        p = next;
    }
}

void ngx_reset_pool(ngx_pool_t* pool)
{
    for (ngx_pool_large_t* l = pool->large; l; l = l->next)
    {
        if (l->alloc)
            ngx_free(l->alloc);
    }

    for (ngx_pool_t* p = pool; p; p = p->d.next)
    {
        p->d.last = (u_char*) p + sizeof(ngx_pool_t);
        p->d.failed = 0;
    }

    pool->current = pool;
    pool->chain = NULL;
    pool->large = NULL;
}

void* ngx_palloc(ngx_pool_t* pool, size_t size) {
    if (size <= pool->max) {
        return ngx_palloc_small(pool, size, 1);
    }
    return ngx_palloc_large(pool, size);
}

void* ngx_pnalloc(ngx_pool_t* pool, size_t size) {
    if (size <= pool->max) {
        return ngx_palloc_small(pool, size, 0);
    }
    return ngx_palloc_large(pool, size);
}

static ngx_inline void* ngx_palloc_small(ngx_pool_t* pool, size_t size, ngx_uint_t align) {
    ngx_pool_t* p = pool->current;
    do {
        u_char* m = p->d.last;
        // align address pointer if required
        if (align != 0) {
            m = ngx_align_ptr(m, NGX_ALIGNMENT);
        }
        // check whether there is enough space left or not
        if ((size_t)(p->d.end - m) >= size) {
            p->d.last = m + size;
            return m;
        }
        p = p->d.next;
    } while (p != NULL);
    return ngx_palloc_block(pool, size);
}

static void* ngx_palloc_block(ngx_pool_t* pool, size_t size) {
    size_t poolSize = (size_t)(pool->d.end - (u_char*)pool);
    u_char* m = ngx_memalign(NGX_POOL_ALIGNMENT, poolSize);
    if (m == NULL) {
        return NULL;
    }

    ngx_pool_t* newPool = (ngx_pool_t*) m;

    newPool->d.end = m + poolSize;
    newPool->d.next = NULL;
    newPool->d.failed = 0;

    m += sizeof(ngx_pool_data_t);
    m = ngx_align_ptr(m, NGX_ALIGNMENT);
    newPool->d.last = m + size;

    // append to the pool list, similar to LRU algorithm
    // so list search steps won't be larger than 6
    ngx_pool_t* p = pool->current;
    for (; p->d.next != NULL; p = p->d.next) {
        if (p->d.failed++ > 4) {
            pool->current = p->d.next;
        }
    }
    p->d.next = newPool;

    return m;
}

static void* ngx_palloc_large(ngx_pool_t* pool, size_t size) {
    void* newBlock = ngx_alloc(size);
    if (newBlock == NULL) {
        return NULL;
    }

    // large object may be deallocated with `ngx_pfree` at any time,
    // and we may reuse a `ngx_pool_large_t` object without creating a new one
    ngx_uint_t n = 0;
    ngx_pool_large_t* large = pool->large;
    for (; large != NULL; large = large->next) {
        if (large->alloc == NULL) {
            large->alloc = newBlock;
            return newBlock;
        }
        // make sure large block list search step won't be longer than 5
        if (n++ > 3) {
            break;
        }
    }

    large = ngx_palloc_small(pool, sizeof(ngx_pool_large_t), 1);
    if (large == NULL) {
        ngx_free(newBlock);
        return NULL;
    }

    large->alloc = newBlock;
    large->next = pool->large;
    pool->large = large;
    return newBlock;
}

void* ngx_pmemalign(ngx_pool_t* pool, size_t size, size_t alignment) {
    void* p = ngx_memalign(alignment, size);
    if (p == NULL) {
        return NULL;
    }

    ngx_pool_large_t* large = ngx_palloc_small(pool, sizeof(ngx_pool_large_t), 1);
    if (large == NULL) {
        ngx_free(p);
        return NULL;
    }

    large->alloc = p;
    large->next = pool->large;
    pool->large = large;
    return p;
}

ngx_int_t ngx_pfree(ngx_pool_t* pool, void* p) {
    for (ngx_pool_large_t* l = pool->large; l != NULL; l = l->next) {
        if (p == l->alloc) {
            err_ret("free: %p", l->alloc);
            ngx_free(l->alloc);
            l->alloc = NULL;
            return NGX_OK;
        }
    }
    return NGX_DECLINED;
}

void* ngx_pcalloc(ngx_pool_t* pool, size_t size) {
    void* p = ngx_palloc(pool, size);
    if (p != NULL) {
        ngx_memzero(p, size);
    }
    return p;
}

ngx_pool_cleanup_t* ngx_pool_cleanup_add(ngx_pool_t* p, size_t size)
{
    ngx_pool_cleanup_t* c = ngx_palloc(p, sizeof(ngx_pool_cleanup_t));
    if (c == NULL)
        return NULL;

    if (size)
    {
        c->data = ngx_palloc(p, size);
        if (c->data == NULL)
            return NULL;
    } else
    {
        c->data = NULL;
    }

    // push front
    c->handler = NULL;
    c->next = p->cleanup;
    p->cleanup = c;
    err_ret("add cleanup: %p", c);
    return c;
}

void ngx_pool_run_cleanup_file(ngx_pool_t* p, ngx_fd_t fd)
{
    for (ngx_pool_cleanup_t* c = p->cleanup; c; c = c->next)
    {
        if (c->handler == ngx_pool_cleanup_file)
        {
            ngx_pool_cleanup_file_t* cf = c->data;
            if (cf->fd == fd)
            {
                c->handler(cf);
                c->handler = NULL;
                return;
            }
        }
    }
}

void ngx_pool_cleanup_file(void *data)
{
    ngx_pool_cleanup_file_t* c = data;

    err_ret("file cleanup: fd:%d", c->fd);

    if (ngx_close_file(c->fd) == NGX_FILE_ERROR)
    {
        err_msg(ngx_close_file_n "\"%s\" failed", c->name);
    }
}

void ngx_pool_delete_file(void *data)
{
    ngx_pool_cleanup_file_t* c = data;

    err_ret("file cleanup: fd:%d %s", c->fd, c->name);

    if (ngx_delete_file(c->name) == NGX_FILE_ERROR)
    {
        ngx_err_t err = ngx_errno;
        if (err != NGX_ENOENT)
        {
            err_msg(ngx_delete_file_n "\"%s\" failed", c->name);
        }
    }

    if (ngx_close_file(c->fd) == NGX_FILE_ERROR)
    {
        err_msg(ngx_close_file_n "\"%s\" failed", c->name);
    }
}
