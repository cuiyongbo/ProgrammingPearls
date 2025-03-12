#include "leetcode.h"

using namespace std;

/* leetcode: 1105 */

class Solution {
public:
    int minHeightShelves(vector<vector<int>>& books, int shelf_width);
};

int Solution::minHeightShelves(vector<vector<int>>& books, int shelf_width) {
/*
    We have a sequence of books: the i-th book has thickness books[i][0] and height books[i][1].
    We want to place these books in order onto bookcase shelves that have total width shelf_width.

    We choose some of the books to place on this shelf (such that the sum of their thickness is <= shelf_width), 
    then build another level of shelf of the bookcase so that the total height of the bookcase has increased by 
    the maximum height of the books we just put down. We repeat this process until there are no more books to place.

    Note again that at each step of the above process, the order of the books we place is the same order 
    as the given sequence of books. For example, if we have an ordered list of 5 books, we might place 
    the first and second book onto the first shelf, the third book on the second shelf, and the fourth and fifth book on the last shelf.

    Return the minimum possible height that the total bookshelf can be after placing shelves in this manner.

    Constraints: books[i][0] <= shelf_width
*/

{ // dp solution
    // dp[i] means minHeightShelves(books[0:i])
    // dp[j] = min(dp[i-1]+max(h[i:j])) for j in [i, n] if sum(w[i:j])<=shelf_width
    int n = books.size();
    vector<int> dp(n, INT32_MAX);
    for (int i=0; i<n; ++i) {
        int h = 0;
        int w = 0;
        int prev = i==0 ? 0 : dp[i-1];
        // put books[i:j] onto the same level of the shief
        for (int j=i; j<n; ++j) {
            w += books[j][0];
            if (w > shelf_width) {
                break;
            }
            h = max(h, books[j][1]);
            dp[j] = min(dp[j], prev+h);
        }
    }
    return dp[n-1];
}

}


void minHeightShelves_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    auto books = stringTo2DArray<int>(input1);
    int actual = ss.minHeightShelves(books, input2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input1, input2, expectedResult, actual);
    }
}


int main() {
    SPDLOG_WARN("Running minHeightShelves tests:");
    TIMER_START(minHeightShelves);
    minHeightShelves_scaffold("[[1,1],[2,3],[2,3],[1,1],[1,1],[1,1],[1,2]]", 4, 6);
    TIMER_STOP(minHeightShelves);
    SPDLOG_WARN("minHeightShelves tests use {} ms", TIMER_MSEC(minHeightShelves));
}
