#include "libmemcached_util.h"

using namespace std;

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

    memcached_server_free(servers);

    memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_BINARY_PROTOCOL, true);

    // The default method is MEMCACHED_DISTRIBUTION_MODULA. 
    // You can enable consistent hashing by setting MEMCACHED_DISTRIBUTION_CONSISTENT.
    // Consistent hashing delivers better distribution and allows servers to be added 
    // to the cluster with minimal cache losses. 
    memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_DISTRIBUTION, MEMCACHED_DISTRIBUTION_CONSISTENT);

    // When enabled a host which is problematic will only be checked for usage 
    // based on the amount of time set by this behavior. The value is in seconds.
    memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_RETRY_TIMEOUT, 20);

    // following options if enabled will remove server from memc
    // memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_REMOVE_FAILED_SERVERS, true) ;  
    // memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_SERVER_FAILURE_LIMIT, 10) ;  // continuous connection retry limit

    const size_t item_count = 4;
    //const char *keys[]= {"0Bl4QH", "psdPY2", "aF0pbU","dG1rms"}; 
    const char *keys[]= {"key1", "key2", "key3","key4"}; 
    const char *values[] = {"This is first value", "This is second value", "This is third value"," This is forth value"}; 

    time_t timeout = 10;
    const char* group_keys[] = {"0Bl4QH", "psdPY2", "aF0pbU","dG1rms"}; 

    int loops = 5;
    while(loops-- > 0)
    {
        for (size_t i = 0; i < item_count; i++)
        {
            // check to see if a key exists
            rc = memcached_exist_by_key(
                    memc, 
                    group_keys[i], strlen(group_keys[i]),
                    keys[i], strlen(keys[i]));
            
            if(rc == MEMCACHED_SUCCESS)
            {
                // retrive key
                size_t value_length = 0;
                uint32_t flags = 0;
                char* value = memcached_get_by_key(
                                memc,
                                group_keys[i], strlen(group_keys[i]),
                                keys[i], strlen(keys[i]),
                                &value_length, &flags, &rc);
                if(value == NULL)
                {
                    cout << "memcached_get_by_key(" << keys[i] << ") failed: " << memcached_strerror(memc, rc) << endl;
                }
                else
                {
                    cout << "memcached_get_by_key(" << keys[i] << ") succeeded: " << value << endl;
                }
                free(value);
            }
            else if(rc == MEMCACHED_NOTFOUND)
            {
                // set key
                rc = memcached_set_by_key(
                        memc, 
                        group_keys[i], strlen(group_keys[i]), 
                        keys[i], strlen(keys[i]),
                        values[i], strlen(values[i]),
                        timeout, (uint32_t)0);

                if(rc == MEMCACHED_SUCCESS)
                {
                    cout << "memcached_set_by_key(" << keys[i] << ") succeeded" << endl;
                }
                else
                {
                    cout << "memcached_set_by_key(" << keys[i] <<") failed: " << memcached_strerror(memc, rc) << endl;
                }
            }
            else
            {
                cout << "memcached_exist_by_key(" << keys[i] << ") failed: " << memcached_strerror(memc, rc) << endl;
            }
        }
        sleep(1 << 2);
    }

    // remove keys
    for (size_t i = 0; i < item_count; i++)
    {
        rc = memcached_delete_by_key(
                memc,
                group_keys[i], strlen(group_keys[i]),
                keys[i], strlen(keys[i]), 0);

        if(rc == MEMCACHED_SUCCESS)
        {
            cout << "memcached_delete_by_key(" << keys[i] << ") succeeded" << endl;
        }
        else
        {
            cout << "memcached_delete_by_key(" << keys[i] <<") failed: " << memcached_strerror(memc, rc) << endl;
        }
    }
    
    memcached_free(memc); 
    return 0; 
}