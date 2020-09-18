#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 102, 107, 199, 1302 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root);
    vector<vector<int>> levelOrder_bottomup(TreeNode* root);
    vector<int> leftSideView(TreeNode* root);
    vector<int> rightSideView(TreeNode* root);
    int deepestLeavesSum(TreeNode* root);
};

int Solution::deepestLeavesSum(TreeNode* root) {
/*
    Given a binary tree, return the sum of values of its deepest leaves.
    Example:
        Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
        Output: 15
*/
    if (root == nullptr) {
        return 0;
    }

    int ans = 0;
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        ans = 0;
        int size = q.size();
        for (int i=0; i<size; i++) {
            auto t = q.front(); q.pop();
            ans += t->val;
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if (t->right != nullptr) {
                q.push(t->right);
            }
        }
    }
    return ans;
}

vector<vector<int>> Solution::levelOrder(TreeNode* root)
{
/*
    Given a binary tree, return the level order traversal of its nodes' values. 
    (ie, from left to right, level by level).
    For example:
    Given binary tree [3,9,20,null,null,15,7],
    return its level order traversal as: [[3],[9,20],[15,7]]
*/

    if (root == nullptr) {
        return vector<vector<int>>();
    }

    vector<int> dummy;
    vector<vector<int>> ans;
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        int size = q.size();
        dummy.clear();
        dummy.reserve(size);
        for (int i=0; i<size; ++i) {
            auto n = q.front(); q.pop();
            dummy.push_back(n->val);
            if(n->left != nullptr) {
                q.push(n->left);
            }
            if (n->right != nullptr) {
                q.push(n->right);
            }
        }
        ans.push_back(dummy);
    }
    return ans;
}

vector<vector<int>> Solution::levelOrder_bottomup(TreeNode* root) {
/*
    Given a binary tree, return the bottom-up level order traversal of its nodes' values. 
    (ie, from left to right, level by level from leaf to root).
*/

    deque<vector<int>> store;
    queue<TreeNode*> q;
    if (root != nullptr) {
        q.push(root);
    } 
    while (!q.empty()) {
        int size = q.size();
        vector<int> dummy;
        dummy.reserve(size);
        for (int i=0; i<size; i++) {
            auto t = q.front(); q.pop();
            dummy.push_back(t->val);
            if(t->left != nullptr) {
                q.push(t->left);
            }
            if(t->right != nullptr) {
                q.push(t->right);
            }
        }
        store.push_front(dummy);
    }
    return vector<vector<int>> (store.begin(), store.end());
}

vector<int> Solution::rightSideView(TreeNode* root) {
/*
    Given a binary tree, imagine yourself standing on the right side of it, 
    return the values of the nodes you can see ordered from top to bottom.
*/

    vector<int> ans;
    queue<TreeNode*> q;
    if (root != NULL) {
        q.push(root);
    }

    while(!q.empty()) {
        int size = q.size();
        for (int i=0; i<size; i++) {
            auto t = q.front(); q.pop();
            if (i+1 == size) {
                ans.push_back(t->val);
            }
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if(t->right != nullptr) {
                q.push(t->right);
            }
        }
    }
    return ans;
}

vector<int> Solution::leftSideView(TreeNode* root) {
/*
    Given a binary tree, imagine yourself standing on the left side of it, 
    return the values of the nodes you can see ordered from top to bottom.
*/
    vector<int> ans;
    queue<TreeNode*> q;
    if (root != NULL) {
        q.push(root);
    }

    while(!q.empty()) {
        int size = q.size();
        for (int i=0; i<size; i++) {
            auto t = q.front(); q.pop();
            if (i == 0) {
                ans.push_back(t->val);
            }
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if(t->right != nullptr) {
                q.push(t->right);
            }
        }
    }
    return ans;
}

void levelOrder_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    auto ans = ss.levelOrder(root);
    auto expected = stringTo2DArray<int>(input2);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
        util::Log(logERROR) << "Actual:";
        for(auto& i: ans) {
            util::Log(logERROR) << numberVectorToString(i);
        }
    }
}

void levelOrder_bottomup_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    auto ans = ss.levelOrder_bottomup(root);
    auto expected = stringTo2DArray<int>(input2);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
        util::Log(logERROR) << "Actual:";
        for(auto& i: ans) {
            util::Log(logERROR) << numberVectorToString(i);
        }
    }
}

void rightSideView_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    auto ans = ss.rightSideView(root);
    auto expected = stringTo1DArray<int>(input2);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
        util::Log(logERROR) << "Actual:" << numberVectorToString(ans);
    }
}

void leftSideView_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    auto ans = ss.leftSideView(root);
    auto expected = stringTo1DArray<int>(input2);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
        util::Log(logERROR) << "Actual:" << numberVectorToString(ans);
    }
}

void deepestLeavesSum_scaffold(string input1, string input2) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    auto ans = ss.deepestLeavesSum(root);
    auto expected = std::stoi(input2);
    if (ans == expected) {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed. Actual: " << ans;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running levelOrder tests:";
    TIMER_START(levelOrder);
    levelOrder_scaffold("[]", "[]");
    levelOrder_scaffold("[1,2,3,4,5]", "[[1],[2,3],[4,5]]");
    levelOrder_scaffold("[3,9,20,null,null,15,7]", "[[3],[9,20],[15,7]]");
    TIMER_STOP(levelOrder);
    util::Log(logESSENTIAL) << "levelOrder using " << TIMER_MSEC(levelOrder) << "ms.";

    util::Log(logESSENTIAL) << "Running levelOrder_bottomup tests:";
    TIMER_START(levelOrder_bottomup);
    levelOrder_bottomup_scaffold("[]", "[]");
    levelOrder_bottomup_scaffold("[1,2,3,4,5]", "[[4,5],[2,3],[1]]");
    levelOrder_bottomup_scaffold("[3,9,20,null,null,15,7]", "[[15,7],[9,20],[3]]");
    TIMER_STOP(levelOrder_bottomup);
    util::Log(logESSENTIAL) << "levelOrder_bottomup using " << TIMER_MSEC(levelOrder_bottomup) << "ms.";

    util::Log(logESSENTIAL) << "Running leftSideView tests:";
    TIMER_START(leftSideView);
    leftSideView_scaffold("[]", "[]");
    leftSideView_scaffold("[1,2,3,4,5]", "[1,2,4]");
    leftSideView_scaffold("[3,9,20,null,null,15,7]", "[3,9,15]");
    TIMER_STOP(leftSideView);
    util::Log(logESSENTIAL) << "leftSideView using " << TIMER_MSEC(leftSideView) << "ms.";

    util::Log(logESSENTIAL) << "Running rightSideView tests:";
    TIMER_START(rightSideView);
    rightSideView_scaffold("[]", "[]");
    rightSideView_scaffold("[1,2,3,4,5]", "[1,3,5]");
    rightSideView_scaffold("[3,9,20,null,null,15,7]", "[3,20,7]");
    TIMER_STOP(rightSideView);
    util::Log(logESSENTIAL) << "rightSideView using " << TIMER_MSEC(rightSideView) << "ms.";    

    util::Log(logESSENTIAL) << "Running deepestLeavesSum tests:";
    TIMER_START(deepestLeavesSum);
    deepestLeavesSum_scaffold("[1,2,3,4,5]", "9");
    deepestLeavesSum_scaffold("[3,9,20,null,null,15,7]", "22");
    deepestLeavesSum_scaffold("[1,2,3,4,5,null,6,7,null,null,null,null,8]", "15");
    TIMER_STOP(deepestLeavesSum);
    util::Log(logESSENTIAL) << "deepestLeavesSum using " << TIMER_MSEC(deepestLeavesSum) << "ms.";
}
