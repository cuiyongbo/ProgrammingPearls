#include "libmemcached_util.h"

using namespace std;

int main()
{
    memcached_return rc; 
    memcached_st* memc = memcached_create(NULL); 

    rc = memcached_server_add_with_weight(memc, "localhost", 11211, 0);
    assert(rc == MEMCACHED_SUCCESS);

    memcached_flush(memc, 0);
    memcached_behavior_set(memc, MEMCACHED_BEHAVIOR_SUPPORT_CAS, true);

    const char* keys[] = {"Venus", NULL};
    const char* values[] = {"Roma", NULL};
    size_t key_lengths[] = {strlen(keys[0]), 0};

    rc = memcached_set(memc, keys[0], strlen(keys[0]), values[0], strlen(values[0]), (time_t)0, (uint32_t)0);
    assert(rc == MEMCACHED_SUCCESS);

    rc = memcached_mget(memc, keys, key_lengths, 1);
    assert(rc == MEMCACHED_SUCCESS);

    memcached_result_st *result = &memc->result;
    result = memcached_fetch_result(memc, result, &rc);
    assert(rc == MEMCACHED_SUCCESS);

    uint64_t cas = memcached_result_cas(result);
    assert(cas > 0);

    assert(memcmp(values[0], memcached_result_value(result), memcached_result_length(result)) == 0);

    values[0] = "Greece";
    rc = memcached_cas(memc, keys[0], strlen(keys[0]), values[0], strlen(values[0]), 0, 0, cas);
    assert(rc == MEMCACHED_SUCCESS);

    rc = memcached_cas(memc, keys[0], strlen(keys[0]), values[0], strlen(values[0]), 0, 0, cas);
    assert(rc == MEMCACHED_DATA_EXISTS);

    memcached_free(memc);
}
