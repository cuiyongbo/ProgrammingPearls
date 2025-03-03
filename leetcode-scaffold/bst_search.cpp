#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 700, 701, 230, 99, 108, 501, 450 */

class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val);
    TreeNode* insertIntoBST(TreeNode* root, int val);
    int kthSmallest(TreeNode* root, int k);
    int kthLargest(TreeNode* root, int k);
    void recoverTree(TreeNode* root);
    TreeNode* sortedArrayToBST(vector<int>& nums);
    vector<int> findMode(TreeNode* root);
    TreeNode* deleteNode(TreeNode* root, int key);
};


/*
    Given the root node of a binary search tree (BST) and a value. 
    You need to find the node in the BST whose value equals the given value. 
    Return the subtree rooted with that node. If such node doesn’t exist, you should return NULL.
*/
TreeNode* Solution::searchBST(TreeNode* root, int val) {
    while (root != nullptr) {
        if (root->val == val) {
            break;
        } else if (root->val > val) {
            root = root->left;
        } else {
            root = root->right;
        }
    }
    return root;
}


/*
    Given the root node of a binary search tree (BST) and a value to be inserted into the tree, 
    insert the value into the BST. Return the root node of the BST after the insertion. 
    It is guaranteed that the new value does not exist in the original BST.
    Note that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. You can return any of them.
*/
TreeNode* Solution::insertIntoBST(TreeNode* root, int val) {
    TreeNode* x = root;
    TreeNode* px = nullptr;
    while (x != nullptr) {
        px = x;
        if (x->val > val) {
            x = x->left;
        } else {
            x = x->right;
        }
    }
    TreeNode* node = new TreeNode(val);
    if (px == nullptr) {
        return node;
    } else {
        if (px->val > val) {
            px->left = node;
        } else {
            px->right = node;
        }
        return root;
    }
}


/*
    Given a binary search tree, write a function to find the kth smallest element in it.
    Note that You may assume k is always valid, 1 ≤ k ≤ BST’s total elements.
    Hint: The answer is the value of kth node when performing inoder traversal
*/
int Solution::kthSmallest(TreeNode* root, int k) {
{ // recursive solution
    int ans = 0;
    int idx = 0; // number of nodes we have traversed so far
    std::function<void(TreeNode*)> inorder_traversal = [&] (TreeNode* node) {
        if (node == nullptr) { // we must omit null node when counting idx
            return;
        }
        inorder_traversal(node->left);
        if (++idx == k) { // you must increase idx when traversing root node
            ans = node->val;
            return; // no need to traverse further nodes
        }
        inorder_traversal(node->right);
    };
    inorder_traversal(root);
    return ans;
}

{ // iterative solution
    int i = 0;
    int ans = 0;
    stack<TreeNode*> st;
    TreeNode* p = root; // left subtree of p to traverse
    while (p!=nullptr || !st.empty()) {
        while (p != nullptr) { // left
            st.push(p);
            p = p->left;
        }
        // st.top()->left == null
        auto t = st.top(); st.pop();
        ans = t->val; // root
        if (++i == k) {
            break;
        }
        p = t->right; // right
    }
    return ans;
}

}


/*
    Given a binary search tree, write a function to find the kth largest element in it.
    Note that You may assume k is always valid, 1 ≤ k ≤ BST’s total elements.
*/
int Solution::kthLargest(TreeNode* root, int k) {
    // what if we need to find the kth largest element in the tree?
    // we can get a sequence in ascending order when we traverse the tree in left->root->right order
    // then we can get a sequence in descending order when we traverse the tree in right->root->left order
    int ans = 0;
    int idx = 0;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        dfs(node->right);
        if (++idx == k) {
            ans = node->val;
            return;
        }
        dfs(node->left);
    };
    dfs(root);
    return ans;
}

/*
    Two elements of a binary search tree (BST) are swapped by mistake. Recover the tree without changing its structure.
    Example 1: 
        Input: [1,3,null,null,2]
        Output: [3,1,null,null,2]
    Example 2: 
        Input: [3,1,4,null,null,2]
        Output: [2,1,4,null,null,3]
    Hint: perform an inorder traversal, then sort the array, and record two elements that have exchanged positions
*/
void Solution::recoverTree(TreeNode* root) {
    TreeNode* p1 = nullptr;
    TreeNode* p2 = nullptr;
    TreeNode* predecessor = nullptr;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        }
        dfs(node->left);
        if (predecessor != nullptr && predecessor->val > node->val) {
            if (p1 == nullptr) {
                p1 = predecessor;
                p2 = node;
            } else {
                p2 = node;
            }
        }
        predecessor = node;
        dfs(node->right);
    };
    dfs(root);
    if (p1 != nullptr && p2 != nullptr) {
        std::swap(p1->val, p2->val);
    }
}


/*
    Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
    For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
*/
TreeNode* Solution::sortedArrayToBST(vector<int>& nums) {
    // l, r are inclusive
    std::function<TreeNode*(int, int)> dfs = [&] (int l, int r) {
        if (l > r) { // trivial case
            return (TreeNode*)nullptr;
        }
        int m = l + (r-l)/2;
        TreeNode* node = new TreeNode(nums[m]);
        node->left = dfs(l, m-1);
        node->right = dfs(m+1, r);
        return node;
    };
    return dfs(0, nums.size()-1);
}


/*
    Given a binary search tree (BST) with duplicates, find all the mode(s) (the most frequently occurred element) in the given BST.
*/
vector<int> Solution::findMode(TreeNode* root) {

if (0) { // naive method
    map<int, int> mp; // node, frequency
    int max_frequency = 0;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* node) {
        if (node==nullptr) {
            return;
        }
        dfs(node->left);
        mp[node->val]++;
        max_frequency = max(max_frequency, mp[node->val]);
        dfs(node->right);
    };
    dfs(root);
    vector<int> ans;
    for (auto n: mp) {
        if (n.second == max_frequency) {
            ans.push_back(n.first);
        }
    }
    return ans;
}

{ // optimal solution
    int cur_freq = 0;
    int max_freq = 0;
    vector<int> ans;
    TreeNode* predecessor = nullptr;
    // traversal subtree rooted at node in in-order
    std::function<void(TreeNode*)> dfs  = [&] (TreeNode* node) {
        if (node == nullptr) {
            return;
        }

        dfs(node->left);

        // we increase cur_freq counter before testing it with max_freq
        // so when the node->val changes, the previous node has already been tested
        if (predecessor != nullptr && predecessor->val == node->val) {
            cur_freq++;
        } else {
            cur_freq = 1;
        }
        if (cur_freq > max_freq) {
            ans.clear();
            ans.push_back(node->val);
            max_freq = cur_freq;
        } else if (cur_freq == max_freq) {
            ans.push_back(node->val);
        }
        predecessor = node;
    
        dfs(node->right);
    };
    dfs(root);
    return ans;
}

}


/*
    Given a root node reference of a BST and a key, delete the node with the given key in the BST. 
    Return the root node reference (possibly updated) of the BST.
    Basically, the deletion can be divided into two stages:
        Search for a node to remove.
        If the node is found, delete the node.
    for example, 
        root:
               4
          1          6
        0   2     5     8
              3 n   n  7  
    key = 0,3,5,7; leaf node; pass
    key = 2,8; node with only one child; pass
    key = 1,6; node with two children; pass
    key = 4; node is root; pass
*/
TreeNode* Solution::deleteNode(TreeNode* root, int key) {

{
    // 1. find the node to be deleted
    TreeNode* x = root;
    TreeNode* px = nullptr; // the parent of x
    while (x != nullptr) {
        if (x->val == key) {
            break;
        } else if (x->val > key) {
            px = x;
            x = x->left;
        } else {
            px = x;
            x = x->right;
        }
    }
    // key is not found, or root is null
    if (x == nullptr) {
        return root;
    }
    // 2. delete the node if found
    // key is one of tree nodes
    if (px == nullptr) { // x is root;
        if (x->left==nullptr || x->right==nullptr) { // at least one child is null
            return x->left!=nullptr ? x->left : x->right;
        } else { // both children are non-null
            // find the successor of x
            TreeNode* xl = x->left;
            TreeNode* xr = x->right;
            TreeNode* xs = x->right;
            TreeNode* pxs = nullptr; // parent of xs
            while (xs->left != nullptr) {
                pxs = xs;
                xs = xs->left;
            }
            assert(xs->left == nullptr);
            if (xs == xr) {
                // replace root with xs
                xs->left = xl;
                return xs;
            } else {
                // replace xs's position with xs->right
                pxs->left = xs->right; xs->right = nullptr;
                // replace root with xs
                xs->left = xl; xs->right = xr;
                return xs;
            }
        }
    } else { // x is non-root
        if (x->left==nullptr && x->right==nullptr) { // x is a leaf node
            if (px->left == x) {
                px->left = nullptr;
            } else {
                px->right = nullptr;
            }
        } else if (x->left==nullptr || x->right==nullptr) { // at least one child is null
            TreeNode* residule = x->left!=nullptr ? x->left : x->right;
            if (px->left == x) {
                px->left = residule;
            } else {
                px->right = residule;
            }
        } else { // both children are non-null
            // find the successor of x
            TreeNode* xl = x->left;
            TreeNode* xr = x->right;
            TreeNode* xs = x->right;
            TreeNode* pxs = nullptr; // parent of xs
            while (xs->left != nullptr) {
                pxs = xs;
                xs = xs->left;
            }
            assert(xs->left == nullptr);
            if (xs == xr) {
                // replace x's position with xs
                if (px->left == x) {
                    px->left = xs;
                } else {
                    px->right = xs;
                }
                xs->left = xl;
            } else {
                // replace xs's position with xs->right
                pxs->left = xs->right; xs->right = nullptr;
                // replace x's position with xs
                if (px->left == x) {
                    px->left = xs;
                } else {
                    px->right = xs;
                }
                xs->left = xl; xs->right = xr;
            }
        }
        return root;
    }
}

{ // solutin from text book
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

    if (x == nullptr) {
        return root;
    } 

    bool rootIsDeleted =  x == root;
    auto transplant_u_with_v = [&] (TreeNode** pu, TreeNode* u, TreeNode* v) {
        if (*pu == nullptr) {
            *pu = v;
        } else if (u == (*pu)->left) {
            (*pu)->left = v;
        } else {
            (*pu)->right = v;
        }
    };

    if (x->left == nullptr) {
        transplant_u_with_v(&px, x, x->right);
    } else if (x->right == nullptr) {
        transplant_u_with_v(&px, x, x->left);
    } else {
        TreeNode* pmxr = x;
        TreeNode* xr = x->right;
        while (xr->left != nullptr) {
            pmxr = xr;
            xr = xr->left;
        }

        if (x->right == xr) {
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

}

void searchBST_scaffold(string input, int val, bool expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    TreeNode* actual = ss.searchBST(root, val);
    if ((actual != nullptr) == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, val, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed", input, val, expectedResult);
    }
}


void insertIntoBST_scaffold(string input, int val, string expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);
    Solution ss;
    TreeNode* actual = ss.insertIntoBST(root, val);
    if (binaryTree_equal(actual, expected)) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, val, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed", input, val, expectedResult);
    }
}


void kthSmallest_scaffold(string input, int k, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    int actual = ss.kthSmallest(root, k);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, k, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, k, expectedResult, actual);
    }
}


void kthLargest_scaffold(string input, int k, int expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    int actual = ss.kthLargest(root, k);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, k, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed, actual: {}", input, k, expectedResult, actual);
    }
}


void recoverTree_scaffold(string input, string expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);
    Solution ss;
    ss.recoverTree(root);
    if (binaryTree_equal(root, expected)) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual:", input, expectedResult);
        printBinaryTree(root);
    }
}


void sortedArrayToBST_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    TreeNode* root = ss.sortedArrayToBST(vi);
    TreeNode* expected = stringToTreeNode(expectedResult);
    if (binaryTree_equal(root, expected)) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual:", input, expectedResult);
        printBinaryTree(root);
    }
}


void findMode_scaffold(string input, string expectedResult) {
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    auto expected = stringTo1DArray<int>(expectedResult);
    vector<int> actual = ss.findMode(root);
    if (actual == expected) {
        SPDLOG_INFO("Case({}, expectedResult={}) passed", input, expectedResult);
    } else {
    SPDLOG_ERROR("Case({}, expectedResult={}) failed. actual: {}", input, expectedResult, numberVectorToString(actual));
    }
}


void deleteNode_scaffold(string input, int k, string expectedResult) {
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);
    Solution ss;
    TreeNode* actual = ss.deleteNode(root, k);
    if (binaryTree_equal(actual, expected)) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed", input, k, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed", input, k, expectedResult);
        cout << "actual: ";
        printBinaryTree(actual);
        cout << "expected: ";
        printBinaryTree(expected);
    }
}


int main() {
    SPDLOG_WARN("Running searchBST tests:");
    TIMER_START(searchBST);
    searchBST_scaffold("[2,1,3]", 1, true);
    searchBST_scaffold("[2,1,3]", 2, true);
    searchBST_scaffold("[2,1,3]", 3, true);
    searchBST_scaffold("[4,2,7,1,3]", 4, true);
    searchBST_scaffold("[4,2,7,1,3]", 2, true);
    searchBST_scaffold("[4,2,7,1,3]", 7, true);
    searchBST_scaffold("[4,2,7,1,3]", 1, true);
    searchBST_scaffold("[4,2,7,1,3]", 3, true);
    searchBST_scaffold("[4,2,7,1,3]", 8, false);
    searchBST_scaffold("[4,2,7,1,3]", 5, false);
    searchBST_scaffold("[4,2,7,1,3]", 0, false);
    TIMER_STOP(searchBST);
    SPDLOG_WARN("searchBST tests use {} ms", TIMER_MSEC(searchBST));

    SPDLOG_WARN("Running insertIntoBST tests:");
    TIMER_START(insertIntoBST);
    insertIntoBST_scaffold("[4,2,7,1,3]", 5, "[4,2,7,1,3,5]");
    insertIntoBST_scaffold("[4,2,7,1,3]", 9, "[4,2,7,1,3,null,9]");
    insertIntoBST_scaffold("[6,2,7,1,3]", 5, "[6,2,7,1,3,null,null,null,null,null,5]");
    TIMER_STOP(insertIntoBST);
    SPDLOG_WARN("insertIntoBST tests use {} ms", TIMER_MSEC(insertIntoBST));

    SPDLOG_WARN("Running kthSmallest tests:");
    TIMER_START(kthSmallest);
    kthSmallest_scaffold("[2,1]", 1, 1);
    kthSmallest_scaffold("[4,2,7,1,3]", 1, 1);
    kthSmallest_scaffold("[6,2,7,1,3]", 3, 3);
    kthSmallest_scaffold("[6,2,7,1,3]", 5, 7);
    kthSmallest_scaffold("[3,1,4,null,2]", 1, 1);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 1, 1);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 2, 2);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 3, 3);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 4, 4);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 5, 5);
    kthSmallest_scaffold("[5,3,6,2,4,null,null,1]", 6, 6);
    TIMER_STOP(kthSmallest);
    SPDLOG_WARN("kthSmallest tests use {} ms", TIMER_MSEC(kthSmallest));

    SPDLOG_WARN("Running kthLargest tests:");
    TIMER_START(kthLargest);
    kthLargest_scaffold("[5,3,6,2,4,null,null,1]", 1, 6);
    kthLargest_scaffold("[5,3,6,2,4,null,null,1]", 2, 5);
    kthLargest_scaffold("[5,3,6,2,4,null,null,1]", 3, 4);
    kthLargest_scaffold("[5,3,6,2,4,null,null,1]", 4, 3);
    kthLargest_scaffold("[5,3,6,2,4,null,null,1]", 5, 2);
    kthLargest_scaffold("[5,3,6,2,4,null,null,1]", 6, 1);
    TIMER_STOP(kthLargest);
    SPDLOG_WARN("kthLargest tests use {} ms", TIMER_MSEC(kthLargest));

    SPDLOG_WARN("Running recoverTree tests:");
    TIMER_START(recoverTree);
    recoverTree_scaffold("[1]", "[1]");
    recoverTree_scaffold("[1,2]", "[2,1]");
    recoverTree_scaffold("[1,2,3]", "[2,1,3]");
    recoverTree_scaffold("[2,3,1]", "[2,1,3]");
    recoverTree_scaffold("[5,3,6,2,1,null,null,4]", "[5,3,6,2,4,null,null,1]");
    recoverTree_scaffold("[5,3,2,6,4,null,null,1]", "[5,3,6,2,4,null,null,1]");
    recoverTree_scaffold("[5,2,6,3,4,null,null,1]", "[5,3,6,2,4,null,null,1]");
    recoverTree_scaffold("[7,1,5,null,null,4,2]", "[2,1,5,null,null,4,7]");
    TIMER_STOP(recoverTree);
    SPDLOG_WARN("recoverTree tests use {} ms", TIMER_MSEC(recoverTree));

    SPDLOG_WARN("Running sortedArrayToBST tests:");
    TIMER_START(sortedArrayToBST);
    sortedArrayToBST_scaffold("[1]", "[1]");
    sortedArrayToBST_scaffold("[1,2]", "[1,null,2]");
    sortedArrayToBST_scaffold("[1,2,3,4,5]", "[3,1,4,null,2,null,5]");
    sortedArrayToBST_scaffold("[-10,-3,0,5,9]", "[0,-10,5,null,-3,null,9]");
    //sortedArrayToBST_scaffold("[1,2]", "[2,1]");
    //sortedArrayToBST_scaffold("[1,2,3,4,5]", "[3,2,5,1,null,4]");
    //sortedArrayToBST_scaffold("[-10,-3,0,5,9]", "[0,-3,9,-10,null,5]");
    TIMER_STOP(sortedArrayToBST);
    SPDLOG_WARN("sortedArrayToBST tests use {} ms", TIMER_MSEC(sortedArrayToBST));

    SPDLOG_WARN("Running findMode tests:");
    TIMER_START(findMode);
    findMode_scaffold("[1,null,2,2]", "[2]");
    findMode_scaffold("[1,2,4,null,null,3,4,null,3]", "[3,4]");
    findMode_scaffold("[1,null,2]", "[1,2]");
    TIMER_STOP(findMode);
    SPDLOG_WARN("findMode tests use {} ms", TIMER_MSEC(findMode));

    SPDLOG_WARN("Running deleteNode tests:");
    TIMER_START(deleteNode);
    deleteNode_scaffold("[5,3,6,2,4,null,7]", 3, "[5,4,6,2,null,null,7]");
    deleteNode_scaffold("[5,3,12,2,4,8,17,null,null,null,null,null,9]", 5, "[8,3,12,2,4,9,17]");
    deleteNode_scaffold("[5,3,6,2,4,null,7]", 5, "[6,3,7,2,4]");
    deleteNode_scaffold("[5,3]", 5, "[3]");
    deleteNode_scaffold("[5,null,6]", 5, "[6]");
    deleteNode_scaffold("[5]", 5, "[]");
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 4, "[5,1,6,0,2,null,8,null,null,null,3,7]"); // root
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 0, "[4,1,6,null,2,5,8,null,3,null,null,7]"); // leaf
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 3, "[4,1,6,0,2,5,8,null,null,null,null,null,null,7]"); // leaf
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 5, "[4,1,6,0,2,null,8,null,null,null,3,7]"); // leaf
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 7, "[4,1,6,0,2,5,8,null,null,null,3]"); // leaf
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 2, "[4,1,6,0,3,5,8,null,null,null,null,null,null,7]"); // node with one non-null child
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 8, "[4,1,6,0,2,5,7,null,null,null,3]"); // node with one non-null child
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 1, "[4,2,6,0,3,5,8,null,null,null,null,null,null,7]"); // node with two non-null children
    deleteNode_scaffold("[4,1,6,0,2,5,8,null,null,null,3,null,null,7]", 6, "[4,1,7,0,2,5,8,null,null,null,3]"); // node with two non-null children
    TIMER_STOP(deleteNode);
    SPDLOG_WARN("deleteNode tests use {} ms", TIMER_MSEC(deleteNode));
}
