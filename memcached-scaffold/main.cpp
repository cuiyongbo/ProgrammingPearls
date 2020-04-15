#include <stdio.h>  
#include <stdlib.h> 
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <libmemcached/memcached.h> 

int main(int argc, char *argv[]) 
{ 
    //connect multi server 
    memcached_return rc; 
    memcached_server_st *servers; 
    servers = memcached_server_list_append(NULL, (char*)"localhost", 11211, &rc); 
    servers = memcached_server_list_append(servers, (char*)"localhost", 30000, &rc); 

    memcached_st* memc = memcached_create(NULL); 
    rc = memcached_server_push(memc, servers);
    memcached_server_free(servers);  
    
    memcached_behavior_set(memc ,MEMCACHED_BEHAVIOR_DISTRIBUTION, MEMCACHED_DISTRIBUTION_CONSISTENT);
    memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_RETRY_TIMEOUT, 20) ;
    // memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_REMOVE_FAILED_SERVERS, true) ;  
    // memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_SERVER_FAILURE_LIMIT, 10) ;

    int times = 0 ;
    int time_sl = 0 ;  
    while(times++<100000)
    {
        //save data 
        const char  *keys[]= {"key1", "key2", "key3","key4"}; 
        const  size_t key_length[]= {4, 4, 4, 4}; 
        char *values[] = {"This is 1 first value", "This is 2 second value", "This is 3 third value"," this is 4 forth value"}; 
        size_t val_length[]= {21, 22, 21, 22};
        for (int i=0; i < 4; i++)      
        {
            rc = memcached_set(memc, keys[i], key_length[i], values[i], val_length[i], (time_t)180, (uint32_t)0);     
            printf("key: %s  rc:%s\n", keys[i], memcached_strerror(memc, rc));   // 输出状态      
        } 
        printf("time: %d\n", time_sl++) ;
        sleep(1) ;
    } 
    
    memcached_free(memc); 
    return 0; 
}