#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercise: 959 */

class Solution 
{
public:
    int regionsBySlashes(const vector<string>& grid);
};

int Solution::regionsBySlashes(const vector<string>& grid)
{
    /*
        In a N x N grid composed of 1 x 1 squares, each 1 x 1 square consists of a /, \, or blank space.  
        These characters divide the square into contiguous regions.

        (Note that backslash characters are escaped, so a \ is represented as "\\".)

        Return the number of regions.

        Hint: split one grid into 4, and use dsu to find unique groups.
        _____________
        |\ /|\ /|\ /|
        |/_\|/_\|/_\|
        |\ /|\ /|\ /|
        |/_\|/_\|/_\|
        |\ /|\ /|\ /|
        |/_\|/_\|/_\|
    */

    int dimension = grid.size();
    DisjointSet dsu(4*dimension*dimension);

    auto merge = [&](int i, int j)
    {
        int s = (i*dimension + j) * 4;

        if(grid[i][j] == '/')
        {
            dsu.unionFunc(s+3, s);
            dsu.unionFunc(s+1, s+2);
        }
        else if(grid[i][j] == '\\')
        {
            dsu.unionFunc(s, s+1);
            dsu.unionFunc(s+2, s+3);
        }
        else
        {
            dsu.unionFunc(s, s+1);
            dsu.unionFunc(s+1, s+2);
            dsu.unionFunc(s+2, s+3);
            dsu.unionFunc(s+3, s);
        }
        
        if(i-1 >= 0)
        {
            dsu.unionFunc(s-4*dimension+2, s);
        }

        if(i+1 < dimension)
        {
            dsu.unionFunc(s+4*dimension, s+2);
        }

        if(j-1 >= 0)
        {
            dsu.unionFunc(s-4+1, s+3);
        }

        if(j+1 < dimension)
        {
            dsu.unionFunc(s+4+3, s+1);
        }
    };

    for(int i=0; i<dimension; i++)
    {
        for(int j=0; j<dimension; j++)
        {
            merge(i, j);
        }
    }

    unordered_set<int> groups;
    for(int i=0; i<4*dimension*dimension; i++)
    {
        groups.emplace(dsu.find(i));
    }
    return groups.size();
}

void regionsBySlashes_scaffold(vector<string> input, int expectedResult)
{
	Solution ss;
	int actual = ss.regionsBySlashes(input);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input[0] << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input[0] << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
	util::LogPolicy::GetInstance().Unmute();

	util::Log(logESSENTIAL) << "Running regionsBySlashes tests:";
	TIMER_START(regionsBySlashes);
	regionsBySlashes_scaffold({" /",  "/ "}, 2);
	regionsBySlashes_scaffold({" /",  "  "}, 1);
	regionsBySlashes_scaffold({"\\/",  "/\\"}, 4);
	regionsBySlashes_scaffold({"/\\",  "\\/"}, 5);
	regionsBySlashes_scaffold({"//",  "/ "}, 3);
	TIMER_STOP(regionsBySlashes);
	util::Log(logESSENTIAL) << "regionsBySlashes using " << TIMER_MSEC(regionsBySlashes) << " milliseconds";
}
