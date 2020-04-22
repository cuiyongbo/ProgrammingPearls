#include "libmemcached_util.h"

using namespace std;

int main(int argc, char *argv[]) 
{ 
    //connect multi-server 
    memcached_return rc; 
    memcached_st* memc = memcached_create(NULL); 
    
    rc = memcached_server_add_with_weight(memc, "localhost", 30000, 10);
    if(rc != MEMCACHED_SUCCESS)
    {
        cout << "memcached_server_add_with_weight: " << memcached_strerror(memc, rc) << endl;
        return 1;
    }

    rc = memcached_server_add_with_weight(memc, "localhost", 11211, 0);
    if(rc != MEMCACHED_SUCCESS)
    {
        cout << "memcached_server_add_with_weight: " << memcached_strerror(memc, rc) << endl;
        return 1;
    }

    uint32_t server_count = memcached_server_count(memc);
    cout << "memcached_server_count: " << server_count << endl;

    const int item_count = 100;
    const int value_length = 10;
    const char* keys[item_count];
    size_t key_length[item_count];

    pairs_st* items = pairs_generate(item_count, value_length);

    for (int i = 0; i < item_count; i++)
    {
        keys[i] = items[i].key;
        key_length[i] = items[i].key_length;
    }
    
    execute_set(memc, items, item_count);

    rc = memcached_mget(memc, keys, key_length, item_count);
    if (rc != MEMCACHED_SUCCESS)
    {
        cout << "memcached_mget failed: " << memcached_strerror(memc, rc) << endl;
    }
    else
    {
        cout << "server:port, key, key_length, value, value_length, flags, cas" << endl;
        memcached_result_st* result = &memc->result;

        while((result = memcached_fetch_result(memc, result, &rc)))
        {
            if(rc == MEMCACHED_SUCCESS)
            {
                memcached_return error;
                const memcached_instance_st* si = memcached_server_by_key(
                        memc, 
                        memcached_result_key_value(result), 
                        memcached_result_key_length(result),
                        &error);
                
                if(si != NULL)
                {
                    cout << memcached_server_name(si) << ":" << memcached_server_port(si) << ", ";   
                }
                else
                {
                   cout << "memcached_server_by_key failed: " << memcached_strerror(memc, error) << endl;
                }

                cout << memcached_result_key_value(result) << ", "
                     << memcached_result_key_length(result) << ", "
                     << memcached_result_value(result) << ", "
                     << memcached_result_length(result) << ", "
                     << memcached_result_flags(result) << ", "
                     << memcached_result_cas(result) << endl;
            }
        }
        assert(rc == MEMCACHED_END);
    }

    pairs_free(items);
    memcached_free(memc); 
    return 0; 
}
