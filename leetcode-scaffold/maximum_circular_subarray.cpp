#include "leetcode.h"

using namespace std;

class Solution {
public:
    int maxSubarraySumCircular(vector<int>& A) {
        int n = A.size();
        long ans1 = helper(A, 0, n-1, 1);
        long s = accumulate(A.begin(), A.end(), 0);
        long ans2 = s + helper(A, 1, n-1, -1);
        long ans3 = s + helper(A, 0, n-2, -1);
        return max(ans1, max(ans2, ans3));
    }
    
private:
    long helper(vector<int>& A, int i, int j, int sign)
    {
        long ans = INT32_MIN, curSum=0;
        for(int k=i; k<=j; ++k)
        {
            curSum = max(curSum, (long)0) + A[k]*sign;
            ans = max(ans, curSum);
        }
        return ans;
    }
};

int main()
{
	vector<int> input {1,-2,3,-2};
	print_vector(input);
	Solution ss;
	cout << ss.maxSubarraySumCircular(input) << "\n";	
	return 0;
}
