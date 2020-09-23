#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 700, 701, 230, 99, 108, 501, 450 */

class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val);
    TreeNode* insertIntoBST(TreeNode* root, int val);
    int kthSmallest(TreeNode* root, int k);
    void recoverTree(TreeNode *root);
    TreeNode* sortedArrayToBST(vector<int>& nums);
    vector<int> findMode(TreeNode* root);
    TreeNode* deleteNode(TreeNode* root, int key);
};

TreeNode* Solution::searchBST(TreeNode* root, int val) {
    while(root != nullptr) {
        if(root->val == val) {
            break;
        } else if(root->val > val) {
            root = root->left;
        } else {
            root = root->right;
        }
    }
    return root;
}

TreeNode* Solution::insertIntoBST(TreeNode* root, int val) {
    TreeNode* t = new TreeNode(val);
    if (root == nullptr) {
        return t;
    }

    TreeNode* p = root;
    TreeNode* q = nullptr;
    while (p != nullptr) {
        q = p;
        if(p->val > val) {
            p = p->left;
        } else {
            p = p->right;
        }
    }

    if (q->val > val) {
        q->left = t;
    }
    else {
        q->right = t;
    }
    return root;
}

int Solution::kthSmallest(TreeNode* root, int k) {
    int ans = 0;
    function<void(TreeNode*)> inorder = [&] (TreeNode* node) {
        if(node == nullptr || k == 0) {
            return;
        }
        inorder(node->left);
        if(--k == 0) {
            ans = node->val;
        }
        inorder(node->right);
    };

    inorder(root);
    return ans;
}

void Solution::recoverTree(TreeNode *root) {
/*
Two elements of a binary search tree (BST) are swapped by mistake.
Recover the tree without changing its structure.
Example 1: 
    Input: [1,3,null,null,2]
    Output: [3,1,null,null,2]
Example 2: 
    Input: [3,1,4,null,null,2]
    Output: [2,1,4,null,null,3]

Straight forward method: perform an inorder traversal, then sort the array, 
and recorde two elements with postions exchanged
*/
    TreeNode* prev = nullptr;
    TreeNode* first = nullptr;
    TreeNode* second = nullptr;

    // perform the inorder traversal
    function<void(TreeNode*)> dfs = [&](TreeNode* node) {
        if(node == nullptr) {
            return;
        }

        dfs(node->left);
        // the sequence must be sorted in ascending order if there is no swap,
        // so if there is a misordered element, record it.
        if(prev != nullptr && prev->val > node->val) {
            if(first == nullptr) {
                first = prev;
            }
            second = node;
        }
        prev = node;
        dfs(node->right);
    };

    dfs(root);

    if(first != nullptr && second != nullptr) {
        swap(first->val, second->val);
    }
}

TreeNode* Solution::sortedArrayToBST(vector<int>& nums)
{
    /*
        Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
        For this problem, a height-balanced binary tree is defined as a binary tree in which the depth
        of the two subtrees of every node never differ by more than 1.
    */

    function<TreeNode*(int, int)> dfs = [&] (int l, int r) {
        if (l>=r) {
            return (TreeNode*)nullptr;
        } else {
            int m = l + (r-l)/2;
            TreeNode* root = new TreeNode(nums[m]);
            root->left = dfs(l, m);
            root->right = dfs(m+1, r);
            return root;
        }
    };
    return dfs(0, nums.size());
}

vector<int> Solution::findMode(TreeNode* root) {
    /*
        Given a binary search tree (BST) with duplicates, 
        find all the mode(s) (the most frequently occurred element) in the given BST.
    */

    vector<int> ans;
    int maxCount=0;
    int curCount=0, curVal=0;
    auto update = [&] (int val) {
        if(curCount > 0 && val == curVal) {
            ++curCount;
        } else {
            curCount = 1;
            curVal = val;
        }
        if(curCount > maxCount) {
            maxCount = curCount;
            ans.clear();
        }
        if(curCount == maxCount) {
            ans.push_back(val);
        }
    };

    function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if(node != nullptr) {
            dfs(node->left);
            update(node->val);
            dfs(node->right);
        }
    };
    dfs(root);
    return ans;
}

TreeNode* Solution::deleteNode(TreeNode* root, int key) {
/*
Given a root node reference of a BST and a key, delete the node with the given key in the BST. 
Return the root node reference (possibly updated) of the BST.
Basically, the deletion can be divided into two stages:
    Search for a node to remove.
    If the node is found, delete the node.
Follow up: Can you solve it with time complexity O(height of tree)?
*/
    // find the node to delete
    TreeNode* x = root;
    TreeNode* px = nullptr;
    while (x != nullptr && x->val != key) {
        px = x;
        if (x->val > key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }

    if(x == nullptr) {
        return root;
    } 

    bool rootIsDeleted =  x == root;
    auto transplant_u_with_v = [&] (TreeNode** pu, TreeNode* u, TreeNode* v) {
        if(*pu == nullptr) {
            *pu = v;
        } else if (u == (*pu)->left) {
            (*pu)->left = v;
        } else {
            (*pu)->right = v;
        }
    };

    if(x->left == nullptr) {
        transplant_u_with_v(&px, x, x->right);
    } else if(x->right == nullptr) {
        transplant_u_with_v(&px, x, x->left);
    } else {
        TreeNode* pmxr = x;
        TreeNode* xr = x->right;
        while(xr->left != nullptr) {
            pmxr = xr;
            xr = xr->left;
        }

        if(x->right == xr) {
            transplant_u_with_v(&px, x, x->right);
            xr->left = x->left;
        } else {
            transplant_u_with_v(&pmxr, xr, xr->right);
            transplant_u_with_v(&px, x, xr);
            xr->left = x->left;
            xr->right = x->right;
        }
    }
    return rootIsDeleted ? px : root;
}

void searchBST_scaffold(string input, int val, bool expectedResult) {
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    TreeNode* actual = ss.searchBST(root, val);
    if((actual != nullptr) == expectedResult) {
        util::Log(logINFO) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") failed";
    }
}

void insertIntoBST_scaffold(string input, int val, string expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    TreeNode* actual = ss.insertIntoBST(root, val);
    if(binaryTree_equal(actual, expected)) {
        util::Log(logINFO) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") failed";
    }
}

void kthSmallest_scaffold(string input, int k, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    int actual = ss.kthSmallest(root, k);
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case (" << input  << ", " << k << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", " << k << ", expected: " << expectedResult << ") failed";
    }
}

void recoverTree_scaffold(string input, string expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    ss.recoverTree(root);
    if(binaryTree_equal(root, expected)) {
        util::Log(logINFO) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }
}

void sortedArrayToBST_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    TreeNode* root = ss.sortedArrayToBST(vi);
    TreeNode* expected = stringToTreeNode(expectedResult);
    if(binaryTree_equal(root, expected)) {
        util::Log(logINFO) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }
}

void findMode_scaffold(string input, string expectedResult) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    auto expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.findMode(root);
    if(actual == expected) {
        util::Log(logINFO) << "Case (" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input << ", " << expectedResult << ") failed, actual: " << numberVectorToString(actual);
    }
}

void deleteNode_scaffold(string input, int k, string expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    TreeNode* actual = ss.deleteNode(root, k);
    if(binaryTree_equal(actual, expected)) {
        util::Log(logINFO) << "Case(" << input << ", " << k << ", " << expectedResult << ") passed"; 
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << k << ", " << expectedResult << ") failed"; 
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running searchBST tests:";
    TIMER_START(searchBST);
    searchBST_scaffold("[1,2,3]", 1, true);
    searchBST_scaffold("[4,2,7,1,3]", 1, true);
    searchBST_scaffold("[4,2,7,1,3]", 8, false);
    searchBST_scaffold("[4,2,7,1,3]", 5, false);
    searchBST_scaffold("[4,2,7,1,3]", 0, false);
    TIMER_STOP(searchBST);
    util::Log(logESSENTIAL) << "searchBST using " << TIMER_MSEC(searchBST) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running insertIntoBST tests:";
    TIMER_START(insertIntoBST);
    insertIntoBST_scaffold("[4,2,7,1,3]", 5, "[4,2,7,1,3,5]");
    insertIntoBST_scaffold("[6,2,7,1,3]", 5, "[6,2,7,1,3,null,null,null,null,null,5]");
    TIMER_STOP(insertIntoBST);
    util::Log(logESSENTIAL) << "insertIntoBST using " << TIMER_MSEC(insertIntoBST) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running kthSmallest tests:";
    TIMER_START(kthSmallest);
    kthSmallest_scaffold("[2,1]", 1, 1);
    kthSmallest_scaffold("[4,2,7,1,3]", 1, 1);
    kthSmallest_scaffold("[6,2,7,1,3]", 3, 3);
    kthSmallest_scaffold("[6,2,7,1,3]", 5, 7);
    kthSmallest_scaffold("[3,1,4,null,2]", 1, 1);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 1, 1);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 3, 3);
    TIMER_STOP(kthSmallest);
    util::Log(logESSENTIAL) << "kthSmallest using " << TIMER_MSEC(kthSmallest) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running recoverTree tests:";
    TIMER_START(recoverTree);
    recoverTree_scaffold("[1]", "[1]");
    recoverTree_scaffold("[1,2]", "[2,1]");
    recoverTree_scaffold("[5,3,6,2,1,null,null,4]", "[5,3,6,2,4,null,null,1]");
    recoverTree_scaffold("[5,3,2,6,4,null,null,1]", "[5,3,6,2,4,null,null,1]");
    recoverTree_scaffold("[5,2,6,3,4,null,null,1]", "[5,3,6,2,4,null,null,1]");
    TIMER_STOP(recoverTree);
    util::Log(logESSENTIAL) << "recoverTree using " << TIMER_MSEC(recoverTree) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running sortedArrayToBST tests:";
    TIMER_START(sortedArrayToBST);
    sortedArrayToBST_scaffold("[1]", "[1]");
    sortedArrayToBST_scaffold("[1,2]", "[2,1]");
    sortedArrayToBST_scaffold("[1,2,3,4,5]", "[3,2,5,1,null,4]");
    sortedArrayToBST_scaffold("[-10,-3,0,5,9]", "[0,-3,9,-10,null,5]");
    TIMER_STOP(sortedArrayToBST);
    util::Log(logESSENTIAL) << "sortedArrayToBST using " << TIMER_MSEC(sortedArrayToBST) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running findMode tests:";
    TIMER_START(findMode);
    findMode_scaffold("[1,null,2,2]", "[2]");
    findMode_scaffold("[1,2,4,null,null,3,4,null,3]", "[3,4]");
    TIMER_STOP(findMode);
    util::Log(logESSENTIAL) << "findMode using " << TIMER_MSEC(findMode) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running deleteNode tests:";
    TIMER_START(deleteNode);
    deleteNode_scaffold("[5,3,6,2,4,null,7]", 3, "[5,4,6,2,null,null,7]");
    deleteNode_scaffold("[5,3,12,2,4,8,17,null,null,null,null,null,9]", 5, "[8,3,12,2,4,9,17]");
    deleteNode_scaffold("[5,3,6,2,4,null,7]", 5, "[6,3,7,2,4]");
    deleteNode_scaffold("[5,3]", 5, "[3]");
    deleteNode_scaffold("[5,null,6]", 5, "[6]");
    deleteNode_scaffold("[5]", 5, "[]");
    TIMER_STOP(deleteNode);
    util::Log(logESSENTIAL) << "deleteNode using " << TIMER_MSEC(deleteNode) << " milliseconds.";
}
