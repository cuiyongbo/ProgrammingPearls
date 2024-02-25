#include <iostream>
#include <unistd.h>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

using namespace std;

void usage()
{
    cout << "Usage: prog loopType[0|1] loopCount nap_in_millisecond\n";
    cout << "\tloopType: \n"; 
    cout << "\t\t0 - serial\n"; 
    cout << "\t\t1 - parallel\n"; 
}

void serial_for_test(int loopCount, int nap_in_millisecond);
void parallel_for_test_classical(int loopCount, int nap_in_millisecond);
void parallel_for_test_lambda(int loopCount, int nap_in_millisecond);

int main(int argc, char* argv[])
{
    std::setbuf(stdin, NULL);
    std::setbuf(stdout, NULL);
    std::setbuf(stderr, NULL);

    if(argc != 4)
    {
        usage();
        return 1;
    }

    int loopType = std::atoi(argv[1]);
    int loopCount = std::atoi(argv[2]);
    int nap_in_millisecond = std::atoi(argv[3]) * 1000;

    if(loopType == 0)
    {
        serial_for_test(loopCount, nap_in_millisecond);
    }
    else
    {
        parallel_for_test_classical(loopCount, nap_in_millisecond);
        //parallel_for_test_lambda(loopCount, nap_in_millisecond);
    }
    
    return 0;
}

void serial_for_test(int loopCount, int nap_in_millisecond)
{
    for(int i=0; i<loopCount; ++i)
    {
        usleep(nap_in_millisecond);
    }
}

void parallel_for_test_classical(int loopCount, int nap_in_millisecond)
{
    class Scaffold
    {
    public:
        Scaffold(int nap): m_interval(nap) {cout << "constructor" << endl;}
        Scaffold(const Scaffold& rhs): m_interval(rhs.m_interval) {cout << "copy constructor" << endl;}
        ~Scaffold() {cout << "destructor" << endl;}

        void operator() (const tbb::blocked_range<std::size_t> &r) const 
        {
            cout << "chunk size: " << r.size() << endl;
            for (std::size_t i = r.begin(); i != r.end(); ++i) 
            {
                usleep(m_interval);
            }
        }

    private:
        int m_interval;
    };

    const int grainSize = 10;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, loopCount, grainSize), Scaffold(nap_in_millisecond));
}

void parallel_for_test_lambda(int loopCount, int nap_in_millisecond)
{
    tbb::blocked_range<size_t> range(0, loopCount, 10);
    cout << "GrainSize: " << range.grainsize() << endl;
    cout << "is_divisible: " << range.is_divisible() << endl;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, loopCount), [&] (const tbb::blocked_range<std::size_t> &r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) 
        {
            usleep(nap_in_millisecond);
        }
    });
}
