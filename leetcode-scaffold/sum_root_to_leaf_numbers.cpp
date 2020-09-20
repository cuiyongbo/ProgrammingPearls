#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 129, 257 */

class Solution {
public:
    int sumNumbers(TreeNode* root);
    vector<string> binaryTreePaths(TreeNode* root);
};

int Solution::sumNumbers(TreeNode* root) {
/*
    Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
    An example is the root-to-leaf path 1->2->3 which represents the number 123.
    Find the total sum of all root-to-leaf numbers.
    Note: A leaf is a node with no children.
*/
    function<int(TreeNode*, int)> dfs = [&](TreeNode* node, int curSum) {
        if(node == nullptr) {
            return 0;
        }
        curSum = curSum * 10 + node->val;
        if (node->left == nullptr && node->right == nullptr) {
            return curSum;
        } else {
            return dfs(node->left, curSum) + dfs(node->right, curSum);
        }
    };

    return dfs(root, 0);
}

vector<string> Solution::binaryTreePaths(TreeNode* root) {
/*
    Given a binary tree, return all root-to-leaf paths.
    Note: A leaf is a node with no children.
*/

    vector<string> ans;
    function<void(TreeNode*,string)> dfs = [&](TreeNode* node, string path) {
        if(node == nullptr) {
            return;
        }

        if(!path.empty()) {
            path += "->";
        }

        path += to_string(node->val);
        if(node->left == nullptr && node->right == nullptr) {
            ans.emplace_back(path);
        } else {
            dfs(node->left, path);
            dfs(node->right, path);
        }
    };

    string path;
    dfs(root, path);
    return ans;
}

void sumNumbers_scaffold(string input, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    int actual = ss.sumNumbers(root);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", " << expectedResult << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed, " 
                            << "expected: " << expectedResult << ", actual: " << actual;
    }
}

void binaryTreePaths_scaffold(string input, vector<string>& expectedResult) {
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    vector<string> actual = ss.binaryTreePaths(root);
    if(actual == expectedResult) {
        util::Log(logESSENTIAL) << "Case(" << input << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ") failed";

        util::Log(logERROR) << "expected:";
        for (auto& s: expectedResult) {
            util::Log(logERROR) << s;
        }
        util::Log(logERROR) << "acutal:";
        for(auto& s: actual) {
            util::Log(logERROR) << s;
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running sumNumbers tests:";
    TIMER_START(sumNumbers);
    sumNumbers_scaffold("[1,2,3]", 25);
    sumNumbers_scaffold("[1,2,3,4,6,null,8]", 388);
    TIMER_STOP(sumNumbers);
    util::Log(logESSENTIAL) << "Running sumNumbers tests uses" << TIMER_MSEC(sumNumbers) << "ms.";

    vector<string> expected;
    expected.emplace_back("1->2");
    expected.emplace_back("1->3");
    binaryTreePaths_scaffold("[1,2,3]", expected);

    expected.clear();
    expected.emplace_back("1->2->4");
    expected.emplace_back("1->2->6");
    expected.emplace_back("1->3->8");
    binaryTreePaths_scaffold("[1,2,3,4,6,null,8]", expected);
}
