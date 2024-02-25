/* LibMemcached
 * Copyright (C) 2006-2009 Brian Aker
 * All rights reserved.
 *
 * Use and distribution licensed under the BSD license.  See
 * the COPYING file in the parent directory for full text.
 *
 * Summary:
 *
 */

#pragma once 

#include <stdio.h>
#include <libmemcached/memcached.h>
#include "random_data_generator.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
    Execute a memcached_set() on a set of pairs.
    Return the number of rows set.
*/
unsigned int execute_set(memcached_st *memc, pairs_st *pairs, unsigned int number_of);

/*
    Execute a memcached_get() on a set of pairs.
    Return the number of rows retrived.
*/
unsigned int execute_get(memcached_st *memc, pairs_st *pairs, unsigned int number_of);

/**
 * Try to run a large mget to get all of the keys
 * @param memc memcached handle
 * @param keys the keys to get
 * @param key_length the length of the keys
 * @param number_of the number of keys to try to get
 * @return the number of keys received
 */
unsigned int execute_mget(memcached_st *memc, const char * const *keys, size_t *key_length,
                          unsigned int number_of);

#ifdef __cplusplus
} // extern "C"
#endif
