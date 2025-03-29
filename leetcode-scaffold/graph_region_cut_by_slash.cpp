#include "leetcode.h"

using namespace std;

/* leetcode: 959 */
class Solution {
public:
    int regionsBySlashes(const vector<string>& grid);
};


/*
    In a N x N grid composed of 1 x 1 squares, each 1 x 1 square consists of a /, \, or blank space.  
    These characters divide the square into contiguous regions. Return the number of regions.

    Hint: split one grid into 4, and use dsu to find unique groups.
    _____________
    |\|/|\|/|\|/|
    |/|\|/|\|/|\|
    |\|/|\|/|\|/|
    |/|\|/|\|/|\|
    -------------
*/
int Solution::regionsBySlashes(const vector<string>& grid) {

    int rows = grid.size();
    int columns = grid[0].size();
    int dimension = grid.size();
    DisjointSet dsu(4*rows*columns);

    auto merge = [&](int i, int j) {
        int s = (i*columns + j) * 4;
        if (grid[i][j] == '/') {
            dsu.unionFunc(s+3, s);
            dsu.unionFunc(s+1, s+2);
        } else if (grid[i][j] == '\\') {
            dsu.unionFunc(s, s+1);
            dsu.unionFunc(s+2, s+3);
        } else {
            dsu.unionFunc(s, s+1);
            dsu.unionFunc(s+1, s+2);
            dsu.unionFunc(s+2, s+3);
        }
        if (i-1 >= 0) {
            dsu.unionFunc(s-4*columns+2, s);
        }
        if (j-1 >= 0) {
            dsu.unionFunc(s-4+1, s+3);
        }
    };

    for(int i=0; i<dimension; i++) {
        for(int j=0; j<dimension; j++) {
            merge(i, j);
        }
    }

    set<int> groups;
    for (int i=0; i<4*dimension*dimension; i++) {
        groups.emplace(dsu.find(i));
    }
    return groups.size();
}


void regionsBySlashes_scaffold(vector<string> input, int expectedResult) {
	Solution ss;
	int actual = ss.regionsBySlashes(input);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input[0], expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed, actual={}", input[0], expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running regionsBySlashes tests:");
	TIMER_START(regionsBySlashes);
	regionsBySlashes_scaffold({" /",  "/ "}, 2);
	regionsBySlashes_scaffold({" /",  "  "}, 1);
	regionsBySlashes_scaffold({"\\/",  "/\\"}, 4);
	regionsBySlashes_scaffold({"/\\",  "\\/"}, 5);
	regionsBySlashes_scaffold({"//",  "/ "}, 3);
	TIMER_STOP(regionsBySlashes);
    SPDLOG_WARN("regionsBySlashes tests use {} ms", TIMER_MSEC(regionsBySlashes));
}