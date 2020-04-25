#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 560*/

class Solution
{
public:
    vector<int> findFrequentTreeSum(TreeNode* root);
};

vector<int> Solution::findFrequentTreeSum(TreeNode* root)
{
    unordered_map<int, int> freqMap;
    function<int(TreeNode*)> dfs = [&](TreeNode* node)
    {
        if(node == NULL) return 0;

        int sum = node->val + dfs(node->left) + dfs(node->right);
        freqMap[sum]++;
        return sum;
    };

    dfs(root);

    struct Element
    {
        int val;
        int freq;

        Element(int v, int f): val(v), freq(f) {}

        bool operator<(const Element& rhs) const
        {
            return std::tie(freq, val) < std::tie(rhs.freq, rhs.val);
        }
    };

    int maxFreq = 0;
    priority_queue<Element, vector<Element>, std::less<Element>> pq;
    for(auto& it: freqMap)
    {
        maxFreq = std::max(maxFreq, it.second);
        pq.emplace(it.first, it.second);
    }

    vector<int> ans;
    while(!pq.empty())
    {
        auto t = pq.top(); pq.pop();
        if(t.freq == maxFreq)
            ans.push_back(t.val);
        else
            break;
    }
    return ans;
}

void findFrequentTreeSum_scaffold(string input, string expected)
{
    TreeNode* root = stringToTreeNode(input);

    vector<int> expectedResult = stringTo1DArray_t<int>(expected);

    Solution ss;
    vector<int> actual = ss.findFrequentTreeSum(root);
    if(actual.size() == expectedResult.size() && equal(actual.begin(), actual.end(), expectedResult.begin()))
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expected << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expected << ") failed";

        util::Log(logERROR) << "expected:";
        for(auto& s: expectedResult)
            util::Log(logERROR) << s;

        util::Log(logERROR) << "acutal:";
        for(auto& s: actual)
            util::Log(logERROR) << s;
    }

    destroyBinaryTree(root);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    findFrequentTreeSum_scaffold("[5,2,-3]", "[4,2,-3]");
    findFrequentTreeSum_scaffold("[5,2,-5]", "[2]");
}
