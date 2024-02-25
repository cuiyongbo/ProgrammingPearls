/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <vector>
#include <list>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>

#include "tbb/parallel_for_each.h"
#include "tbb/parallel_sort.h"
#include "tbb/tick_count.h"

template<typename T>
void foo(T& f)
{
    f+= 1;
}

template<typename Container>
void test(std::string test_name, int N, int repeat_count)
{
    typedef typename Container::value_type Type;

    Container v;
    for(int i=0; i<N; ++i)
    {
        v.push_back(static_cast<Type>(std::rand()));
    }

    std::vector<double> times(repeat_count, 0);
    for (int i = 0; i < repeat_count; ++i)
    {
        tbb::tick_count t0 = tbb::tick_count::now();
        tbb::parallel_for_each(v.begin(), v.end(), foo<Type>);
        tbb::tick_count t1 = tbb::tick_count::now();
        times[i] = ((t1 - t0).seconds() * 1e03);
    }
    
    // std::sort(times.begin(), times.end());
    //tbb::parallel_sort(times.begin(), times.end());
    tbb::parallel_sort(times);
    double median = times[repeat_count/2];
    if(repeat_count % 2 == 0)
    {
        median = (median + times[repeat_count/2-1]) * 0.5;
    }

    std::cout << "Test: " << test_name << std::endl
            << "\tmin: " << times[0] << " ms" << std::endl
            << "\tmedian: " << median << " ms" << std::endl
            << "\tmax: " << times[repeat_count-1] << " ms" << std::endl;
}

int main(int argc, char* argv[])
{
    int N = argc > 1 ? std::atoi(argv[1]) : 1000;
    int repeat_count = argc > 2 ? std::atoi(argv[2]) : 10;

    srand(0);
    test<std::vector<int>>("std::vector<int>", N, repeat_count);
    test<std::list<float>>("std::list<float>", N, repeat_count);

    return 0;
}
