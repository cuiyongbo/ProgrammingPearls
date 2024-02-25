#include "leetcode.h"

using namespace std;
using namespace osrm;

/*leetcode: 102, 107, 199, 662, 1302 */
class Solution {
public:
    std::vector<std::vector<int>> levelOrder(TreeNode* root);
    std::vector<std::vector<int>> levelOrder_107(TreeNode* root);
    std::vector<int> leftSideView(TreeNode* root);
    std::vector<int> rightSideView(TreeNode* root);
    int widthOfBinaryTree(TreeNode* root);
    int deepestLeavesSum(TreeNode* root);
};

int Solution::widthOfBinaryTree(TreeNode* root) {
/*
    Given the root of a binary tree, return the maximum width of the given tree.
    The maximum width of a tree is the maximum width among all levels.
    The width of one level is defined as the length between the end-nodes (the leftmost and 
    rightmost non-null nodes), where the null nodes between the end-nodes are also taken into calculation.
*/

{
    int width = 0;
    std::queue<std::pair<TreeNode*, long>> q; // node, node_id, for node at i (1-indexed), the node_ids of its children is 2i, 2i+1
    if (root != nullptr) {
        q.push(make_pair(root, 1));
    }
    while (!q.empty()) {
        int offset = 0; // minus offset when calculating node_id to prevent integer range overflow
        int left=-1, right=-1;
        int sz = q.size();
        for(int i=0; i<sz; ++i) {
            auto p = q.front(); q.pop();

            // 1. at each loop, we calculate the width of last level
            // the boundaries are easy to be decided since nodes in the queue are non-null
            if (left == -1) {
                left = p.second;
                offset = left;
            }
            right = p.second;

            // If we index nodes (1-indexed) by level from left to right, then if a node is labelled by i, its left child is 2i, and right child is 2i+1,
            // Further more, we substract the base offset at each level, which is 2**(level-1) if the tree is complete.
            if (p.first->left != nullptr) {
                q.push(make_pair(p.first->left, (p.second - offset)*2));
            }
            if (p.first->right != nullptr) {
                q.push(make_pair(p.first->right, (p.second - offset)*2+1));
            }
        }
        width = max(width, right-left+1);
        //printf("level: %d, left: %d, right: %d, width: %d, offset: %d\n", level++, left, right, width, offset);
    }
    return width;
}

}

int Solution::deepestLeavesSum(TreeNode* root) {
/*
    Given a binary tree, return the sum of values of its deepest leaves.
    Example:
        Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
        Output: 15
        Explanation: the deepest leaves are [7, 8]
*/
    int ans = 0;
    std::queue<TreeNode*> q;
    if (root != nullptr) {
        q.push(root);   
    }
    while (!q.empty()) {
        ans = 0;
        for (int k=q.size(); k!=0; --k) {
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

std::vector<std::vector<int>> Solution::levelOrder(TreeNode* root) {
/*
    Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
    For example: Given binary tree [3,9,20,null,null,15,7], return its level order traversal as: [[3],[9,20],[15,7]]
*/
    std::vector<int> dummy;
    std::vector<std::vector<int>> ans;
    std::queue<TreeNode*> q;
    if (root != nullptr) {
        q.push(root);
    }
    while (!q.empty()) {
        dummy.clear();
        for (int k=q.size(); k>0; --k) {
            auto t = q.front(); q.pop();
            dummy.push_back(t->val);
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if (t->right != nullptr) {
                q.push(t->right);
            }
        }
        ans.push_back(dummy);
    }
    return ans;
}

std::vector<std::vector<int>> Solution::levelOrder_107(TreeNode* root) {
/*
    Given a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).
*/

    std::vector<int> dummy;
    std::vector<std::vector<int>> ans;
    std::queue<TreeNode*> q;
    if (root != nullptr) {
        q.push(root);
    }
    while (!q.empty()) {
        dummy.clear();
        for (int k=q.size(); k>0; --k) {
            auto t = q.front(); q.pop();
            dummy.push_back(t->val);
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if (t->right != nullptr) {
                q.push(t->right);
            }
        }
        ans.push_back(dummy);
    }
    std::reverse(ans.begin(), ans.end());
    return ans;
}

std::vector<int> Solution::rightSideView(TreeNode* root) {
/*
    Given a binary tree, imagine yourself standing on the right side of it, 
    return the values of the nodes you can see ordered from top to bottom.
*/

    std::vector<int> ans;
    std::queue<TreeNode*> q;
    if (root != nullptr) {
        q.push(root);
    }
    while (!q.empty()) {
        for (int i=q.size(); i!=0; --i) {
            auto t = q.front(); q.pop();
            if (i == 1) {
                ans.push_back(t->val);
            }
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

std::vector<int> Solution::leftSideView(TreeNode* root) {
/*
    Given a binary tree, imagine yourself standing on the left side of it, 
    return the values of the nodes you can see ordered from top to bottom.
*/
    std::vector<int> ans;
    std::queue<TreeNode*> q;
    if (root != nullptr) {
        q.push(root);
    }
    while (!q.empty()) {
        int size = q.size();
        for (int i=0; i<size; ++i) {
            auto t = q.front(); q.pop();
            if (i == 0) {
                ans.push_back(t->val);
            }
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

void levelOrder_scaffold(std::string input1, std::string input2, int func_no) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    std::vector<std::vector<int>> ans;
    if (func_no == 102) {
        ans = ss.levelOrder(root);
    } else if (func_no == 107) {
        ans = ss.levelOrder_107(root);
    } else {
        util::Log(logERROR) << "func_no can only be values in [102, 107]";
        return;
    }
    auto expected = stringTo2DArray<int>(input2);
    if (ans == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << func_no << ") passed.";
    } else {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << func_no << ") failed. actual:";
        for(auto& i: ans) {
            util::Log(logERROR) << numberVectorToString(i);
        }
    }
}

void sideView_scaffold(std::string input1, std::string input2, bool left) {
    TreeNode* root = stringToTreeNode(input1);
    std::vector<int> ans;
    Solution ss;
    if (left) {
        ans = ss.leftSideView(root);
    } else {
        ans = ss.rightSideView(root);
    }
    auto expected = stringTo1DArray<int>(input2);
    if (ans == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << (left ? "left" : "right") << ") passed.";
    } else {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << (left ? "left" : "right") << ") failed. actual:" << numberVectorToString(ans);
    }
}

void deepestLeavesSum_scaffold(std::string input1, std::string input2) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    auto ans = ss.deepestLeavesSum(root);
    auto expected = std::stoi(input2);
    if (ans == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed. Actual: " << ans;
    }
}

void widthOfBinaryTree_scaffold(std::string input1, int expected) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    int ans = ss.widthOfBinaryTree(root);
    if (ans == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expected << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expected << ") failed, "
                            << "actual: " << ans << ", expected: " << expected;
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running levelOrder tests:";
    TIMER_START(levelOrder);
    levelOrder_scaffold("[]", "[]", 102);
    levelOrder_scaffold("[1,2,3,4,5]", "[[1],[2,3],[4,5]]", 102);
    levelOrder_scaffold("[3,9,20,null,null,15,7]", "[[3],[9,20],[15,7]]", 102);
    levelOrder_scaffold("[]", "[]", 107);
    levelOrder_scaffold("[1,2,3,4,5]", "[[4,5],[2,3],[1]]", 107);
    levelOrder_scaffold("[3,9,20,null,null,15,7]", "[[15,7],[9,20],[3]]", 107);
    TIMER_STOP(levelOrder);
    util::Log(logESSENTIAL) << "levelOrder using " << TIMER_MSEC(levelOrder) << "ms.";

    util::Log(logESSENTIAL) << "Running SideView tests:";
    TIMER_START(SideView);
    sideView_scaffold("[]", "[]", true);
    sideView_scaffold("[1,2,3,4,5]", "[1,2,4]", true);
    sideView_scaffold("[3,9,20,null,null,15,7]", "[3,9,15]", true);
    sideView_scaffold("[]", "[]", false);
    sideView_scaffold("[1,2,3,4,5]", "[1,3,5]", false);
    sideView_scaffold("[3,9,20,null,null,15,7]", "[3,20,7]", false);
    TIMER_STOP(SideView);
    util::Log(logESSENTIAL) << "SideView using " << TIMER_MSEC(SideView) << "ms.";

    util::Log(logESSENTIAL) << "Running deepestLeavesSum tests:";
    TIMER_START(deepestLeavesSum);
    deepestLeavesSum_scaffold("[1,2,3,4,5]", "9");
    deepestLeavesSum_scaffold("[3,9,20,null,null,15,7]", "22");
    deepestLeavesSum_scaffold("[1,2,3,4,5,null,6,7,null,null,null,null,8]", "15");
    TIMER_STOP(deepestLeavesSum);
    util::Log(logESSENTIAL) << "deepestLeavesSum using " << TIMER_MSEC(deepestLeavesSum) << "ms.";

    util::Log(logESSENTIAL) << "Running widthOfBinaryTree tests:";
    TIMER_START(widthOfBinaryTree);
    widthOfBinaryTree_scaffold("[]", 0);
    widthOfBinaryTree_scaffold("[1]", 1);
    widthOfBinaryTree_scaffold("[1,3,2,5,3,null,9]", 4);
    widthOfBinaryTree_scaffold("[1,3,null,5,3]", 2);
    widthOfBinaryTree_scaffold("[1,2,3,4,5]", 2);
    widthOfBinaryTree_scaffold("[1,3,2,5]", 2);
    widthOfBinaryTree_scaffold("[1,3,2,5,null,null,9,6,null,null,7]", 8);
    widthOfBinaryTree_scaffold("[1,3,2,5,null,null,9,6,null,7]", 7);
    widthOfBinaryTree_scaffold("[1,3,2,5,null,null,9,null,6,7]", 6);
    widthOfBinaryTree_scaffold("[1,1,1,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,null,1,1,null,1,null,1,null,1,null,1,null]", 2147483645);
    widthOfBinaryTree_scaffold("[1,1,1,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,null,1,1,null,1,null,1,null,1,null,1,null,null,null,1]", 2147483645);
    TIMER_STOP(widthOfBinaryTree);
    util::Log(logESSENTIAL) << "widthOfBinaryTree tests using " << TIMER_MSEC(widthOfBinaryTree) <<"ms";
}
