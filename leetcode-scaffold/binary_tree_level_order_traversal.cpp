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
    if (root == nullptr) {
        return 0;
    }
    int width = 0;
    // for a node with node_id i (0-indexed), left(i) = 2*i+1, right(i) = 2*i+2
    using element_t = std::pair<TreeNode*, int>; // node, node_id
    queue<element_t> q; q.push({root, 0});
    while (!q.empty()) {
        int offset = 0;
        int left=-1, right=-1;
        for (int k=q.size(); k!=0; k--) {
            auto t = q.front(); q.pop();
            if (left == -1) {
                left = t.second;
                offset = t.second; // minus offset when calculating node_id to prevent integer overflow
            }
            right = t.second;
            if (t.first->left != nullptr) {
                q.push({t.first->left, 2*(t.second - offset)+1});
            }
            if (t.first->right != nullptr) {
                q.push({t.first->right, 2*(t.second - offset)+2});
            }
        }
        width = max(width, right-left+1);
    }
    return width;
}


int Solution::deepestLeavesSum(TreeNode* root) {
/*
    Given a binary tree, return the sum of values of its deepest leaves.
    Example:
        Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
        Output: 15
        Explanation: the deepest leaves are [7, 8]
*/
    if (root == nullptr) {
        return 0;
    }
    int ans = 0;
    std::queue<TreeNode*> q; q.push(root);   
    while (!q.empty()) {
        ans = 0; // reset `ans` before starting traversing each level
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
    std::vector<std::vector<int>> ans;
    if (root == nullptr) {
        return ans;
    }
    std::vector<int> buffer;
    std::queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        buffer.clear();
        for (int k=q.size(); k!=0; --k) {
            auto t = q.front(); q.pop();
            buffer.push_back(t->val);
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if (t->right != nullptr) {
                q.push(t->right);
            }
        }
        ans.push_back(buffer);
    }
    return ans;
}


std::vector<std::vector<int>> Solution::levelOrder_107(TreeNode* root) {
/*
    Given a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).
*/
    std::vector<std::vector<int>> ans;
    if (root == nullptr) {
        return ans;
    }
    std::vector<int> buffer;
    std::queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        buffer.clear();
        for (int k=q.size(); k!=0; --k) {
            auto t = q.front(); q.pop();
            buffer.push_back(t->val);
            if (t->left != nullptr) {
                q.push(t->left);
            }
            if (t->right != nullptr) {
                q.push(t->right);
            }
        }
        ans.push_back(buffer);
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
        SPDLOG_ERROR("func_no can only be values in [102, 107], actual: {}", func_no);
        return;
    }
    auto expected = stringTo2DArray<int>(input2);
    if (ans == expected) {
        SPDLOG_INFO("Case({}, {}, func_no={}) passed", input1, input2, func_no);
    } else {
        SPDLOG_ERROR("Case({}, {}, func_no={}) failed. actual:", input1, input2, func_no);
        for(auto& i: ans) {
            std::cout << numberVectorToString(i) << std::endl;;
        }
        std::cout << std::endl;
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
        SPDLOG_INFO("Case({}, expectedResult={}, {}) passed", input1, input2, (left ? "left" : "right"));
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}, {}) failed. actual: {}", input1, input2, (left ? "left" : "right"), numberVectorToString(ans));
    }
}


void deepestLeavesSum_scaffold(std::string input1, int expected) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    int ans = ss.deepestLeavesSum(root);
    if (ans == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expected);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input1, expected, ans);
    }
}


void widthOfBinaryTree_scaffold(std::string input1, int expected) {
    TreeNode* root = stringToTreeNode(input1);
    Solution ss;
    int ans = ss.widthOfBinaryTree(root);
    if (ans == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input1, expected);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input1, expected, ans);
    }
}


int main() {
    SPDLOG_WARN("Running levelOrder tests:");
    TIMER_START(levelOrder);
    levelOrder_scaffold("[]", "[]", 102);
    levelOrder_scaffold("[1,2,3,4,5]", "[[1],[2,3],[4,5]]", 102);
    levelOrder_scaffold("[3,9,20,null,null,15,7]", "[[3],[9,20],[15,7]]", 102);
    levelOrder_scaffold("[]", "[]", 107);
    levelOrder_scaffold("[1,2,3,4,5]", "[[4,5],[2,3],[1]]", 107);
    levelOrder_scaffold("[3,9,20,null,null,15,7]", "[[15,7],[9,20],[3]]", 107);
    TIMER_STOP(levelOrder);
    SPDLOG_WARN("levelOrder using {} ms", TIMER_MSEC(levelOrder));

    SPDLOG_WARN("Running SideView tests:");
    TIMER_START(SideView);
    sideView_scaffold("[]", "[]", true);
    sideView_scaffold("[1,2,3,4,5]", "[1,2,4]", true);
    sideView_scaffold("[3,9,20,null,null,15,7]", "[3,9,15]", true);
    sideView_scaffold("[]", "[]", false);
    sideView_scaffold("[1,2,3,4,5]", "[1,3,5]", false);
    sideView_scaffold("[3,9,20,null,null,15,7]", "[3,20,7]", false);
    TIMER_STOP(SideView);
    SPDLOG_WARN("SideView using {} ms", TIMER_MSEC(SideView));

    SPDLOG_WARN("Running deepestLeavesSum tests:");
    TIMER_START(deepestLeavesSum);
    deepestLeavesSum_scaffold("[1,2,3,4,5]", 9);
    deepestLeavesSum_scaffold("[3,9,20,null,null,15,7]", 22);
    deepestLeavesSum_scaffold("[1,2,3,4,5,null,6,7,null,null,null,null,8]", 15);
    TIMER_STOP(deepestLeavesSum);
    SPDLOG_WARN("deepestLeavesSum using {} ms", TIMER_MSEC(deepestLeavesSum));

    SPDLOG_WARN("Running widthOfBinaryTree tests:");
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
    SPDLOG_WARN("widthOfBinaryTree using {} ms", TIMER_MSEC(widthOfBinaryTree));
}
