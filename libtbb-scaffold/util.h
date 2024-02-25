#pragma once
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <unistd.h>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_for_each.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"

inline void set_stdout_no_buffering() {
    std::setbuf(stdout, NULL);
    std::setbuf(stderr, NULL);
}
