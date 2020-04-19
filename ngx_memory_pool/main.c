#include "ngx_core.h"

// compile with ``gcc main.c ngx_alloc.c ngx_palloc.c -o scaffold``

int main()
{
    ngx_init_mem_params();

    printf("memory allocation test: \n");
    const char* s1 = "hello world";
    const char* s2 = "hello world";
    void* p = ngx_alloc(20);
    memcpy(p, s1, strlen(s1));
    printf("p: %s\n", (char*)p);
    printf("s1 address: %p\n", s1);
    printf("s2 address: %p\n", s2);
    ngx_free(p);
    p = NULL;

    printf("memory pool test: \n");
    ngx_pool_t* pool = ngx_create_pool(1024);
    for(int i=0; i<20; ++i)
    {
        void* p = ngx_palloc(pool, 1<<i);
        printf("address: %p\n", p);
    }
    ngx_destroy_pool(pool);

    return 0;
}