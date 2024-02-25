#include "libmemcached_util.h"

using namespace std;

// server traversal

int main(int argc, char *argv[]) 
{ 
    //connect multi-server 
    memcached_return rc; 
    memcached_server_st *servers = NULL; 
    servers = memcached_server_list_append(servers, (char*)"localhost", 30000, &rc); 
    servers = memcached_server_list_append(servers, (char*)"localhost", 11211, &rc); 

    memcached_st* memc = memcached_create(NULL); 
    rc = memcached_server_push(memc, servers);
    if(rc != MEMCACHED_SUCCESS)
    {
        cout << "memcached_server_push: " << memcached_strerror(memc, rc) << endl;
        return 1;
    }

    uint32_t server_count = memcached_server_count(memc);
    cout << "memcached_server_count: " << server_count << endl;

    print_version(memc);

    memcached_server_free(servers);
    memcached_free(memc); 
    return 0; 
}
