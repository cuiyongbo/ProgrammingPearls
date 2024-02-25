#include <iostream>
#include <unistd.h>

#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"

using namespace std;

class Scaffold 
{
public:
    void operator() (const tbb::blocked_range2d<int> &r) const
    {
        for (int y = r.rows().begin(); y != r.rows().end(); ++y) 
        {
            for (int x = r.cols().begin(); x != r.cols().end(); x++) 
            {
                cout << x << ", " << y << endl;
            }
        }
    }
};

int main()
{
    std::setbuf(stdout, NULL);
    std::setbuf(stderr, NULL);

    tbb::parallel_for(tbb::blocked_range2d<int>(0, 10, 0, 10), Scaffold());
}
