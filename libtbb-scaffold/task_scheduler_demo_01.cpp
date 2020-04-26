#include "util.h"

using namespace std;

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        cout << "Usage: prog thread_number" << endl;
        return 1;
    }

    int thread_number = std::atoi(argv[1]);
    tbb::task_scheduler_init init(thread_number);
    assert(init.is_active());

    int loopCount = 10000;
    tbb:parallel_for(tbb::blocked_range<int>(0, loopCount), [](const tbb::blocked_range<int>& r)
    {
        for (int i = r.begin(); i != r.end(); i++)
        {
            usleep(1000);
        }
    });
    return 0;
}