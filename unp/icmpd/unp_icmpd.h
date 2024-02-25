#pragma once

#include "unp.h"

#define ICMPD_PATH "/tmp/icmpd"

struct Icmpd_error
{
    int icmpd_errno; // EHOSTUNREACH, EMSGSIZE, ECONNREFUSED
    int icmpd_type;  // actual ICMPv[46] type
    int icmpd_code;  // actual ICMPv[46] code
    socklen_t icmpd_len;
    struct sockaddr_storage icmpd_dest;
};
