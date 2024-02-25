#include "leetcode.h"

using namespace std;
using namespace osrm;

/*
   Given a rod of length n inches and a table of prices :math:`p_i \text{ for i } \in [1,n]`, 
   determine the maximum revenue :math:`r_n` obtainable by cutting up the rod and selling the pieces.
*/

class Solution {
public:
    int cut_rod(vector<int>& price_list, int rod_len);
    // optimal revenue, cut plan
    pair<int, string> extend_cut_rod(vector<int>& price_list, int rod_len);
};

int Solution::cut_rod(vector<int>& price_list, int rod_len) {
    vector<int> revenue(rod_len+1, INT32_MIN);
    revenue[0] = 0;
    for (int i=1; i<=rod_len; ++i) {
        int q = INT32_MIN;
        for (int j=1; j<=i; ++j) {
            q = max(q, price_list[j]+revenue[i-j]);
        }
        revenue[i] = q;
    }
    return revenue[rod_len];
}

pair<int, string> Solution::extend_cut_rod(vector<int>& price_list, int rod_len) {
    vector<int> revenue(rod_len+1, INT32_MIN);
    vector<int> plan(rod_len+1, 0);
    revenue[0] = 0;
    for (int i=1; i<=rod_len; ++i) {
        int q = INT32_MIN;
        for (int j=1; j<=i; ++j) {
            if (q < price_list[j]+revenue[i-j]) {
                q = price_list[j]+revenue[i-j];
                plan[i] = j;
            }
        }
        revenue[i] = q;
    }
    string ans;
    int n = rod_len;
    while (n > 0) {
        ans.append(std::to_string(plan[n]));
        ans.append(",");
        n -= plan[n];
    }
    if (!ans.empty()) {
        ans.pop_back();
    } else {
        ans = "nil";
    }
    cout << "case(" << numberVectorToString(price_list) << ", " <<  rod_len << "): " 
         << revenue[rod_len] << ", " << ans << endl; 
    return std::make_pair(revenue[rod_len], ans);
}

void basic_test() {
    // price_list[i] means the price of rod with length i is price_list[i]
    vector<int> price_list {
        0,1,5,8,9,10,17,17,20 // 0-8
    };

    Solution ss;
    for (int i=0; i<price_list.size(); ++i) {
        ss.extend_cut_rod(price_list, i);
    }
}

int main() {
    basic_test();
}