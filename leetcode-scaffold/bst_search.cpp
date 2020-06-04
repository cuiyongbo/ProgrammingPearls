#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 700, 701, 230, 99, 108, 501, 450 */

class Solution
{
public:
    TreeNode* searchBST(TreeNode* root, int val);
    TreeNode* insertIntoBST(TreeNode* root, int val);
    int kthSmallest(TreeNode* root, int k);
    void recoverTree(TreeNode *root);
    TreeNode* sortedArrayToBST(vector<int>& nums);
    vector<int> findMode(TreeNode* root);
    TreeNode* deleteNode(TreeNode* root, int key);
    TreeNode* minimum(TreeNode* root);

private:
    void recoverTree_recursive(TreeNode*);
    void recoverTree_array_failure(TreeNode*);
    TreeNode* sortedArrayToBST_workhorse(vector<int>& nums, int left, int right);
};

TreeNode* Solution::searchBST(TreeNode* root, int val)
{
    while(root != NULL)
    {
        if(root->val == val)
        {
            break;
        }
        else if(root->val > val)
        {
            root = root->left;
        }
        else
        {
            root = root->right;
        }
    }
    return root;
}

TreeNode* Solution::insertIntoBST(TreeNode* root, int val)
{
    TreeNode* t = new TreeNode(val);
    if(root == NULL) return t;

    TreeNode* p = root;
    TreeNode* q = NULL;
    while(p != NULL)
    {
        q = p;
        if(p->val > val)
        {
            p = p->left;
        }
        else
        {
            p = p->right;
        }
    }

    if(q->val > val)
        q->left = t;
    else
        q->right = t;

    return root;
}

int Solution::kthSmallest(TreeNode* root, int k)
{
    int ans = 0;
    function<void(TreeNode*)> inorder = [&] (TreeNode* node)
    {
        if(node == NULL || k == 0)
            return;

        inorder(node->left);

        if(--k == 0)
        {
            ans = node->val;
        }

        inorder(node->right);
    };

    inorder(root);
    return ans;
}

void Solution::recoverTree(TreeNode* root)
{
    return recoverTree_recursive(root);
    // return recoverTree_array(root);
}

void Solution::recoverTree_recursive(TreeNode *root)
{
    TreeNode* prev = NULL;
    TreeNode* first = NULL;
    TreeNode* second = NULL;
    function<void(TreeNode*)> dfs = [&](TreeNode* node)
    {
        if(node == NULL)
            return;

        dfs(node->left);

        if(prev != NULL && prev->val > node->val)
        {
            if(first == NULL)
            {
                first = prev;
            }

            second = node;
        }

        prev = node;

        dfs(node->right);
    };

    dfs(root);

    if(first != NULL && second != NULL)
        swap(first->val, second->val);
}

void Solution::recoverTree_array_failure(TreeNode* root)
{
    vector<TreeNode*> inorderSeq;
    function<void(TreeNode*)> dfs = [&](TreeNode* node)
    {
        if(node == NULL) return;

        dfs(node->left);
        inorderSeq.push_back(node);
        dfs(node->right);
    };

    dfs(root);

    TreeNode* first = NULL;
    TreeNode* second = NULL;

    int i = 0;
    int size = inorderSeq.size();
    for(; first == NULL && i<size-1; ++i)
    {
        if(inorderSeq[i]->val > inorderSeq[i+1]->val)
            first = inorderSeq[i];
    }

    for(; second == NULL && i < size; ++i)
    {
        if(inorderSeq[i-1] > inorderSeq[i])
            second = inorderSeq[i];
    }

    if(first != NULL && second != NULL)
        swap(first->val, second->val);
}

TreeNode* Solution::sortedArrayToBST(vector<int>& nums)
{
    /*
        Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
        For this problem, a height-balanced binary tree is defined as a binary tree in which the depth
        of the two subtrees of every node never differ by more than 1.
    */
    return sortedArrayToBST_workhorse(nums, 0, nums.size());
}

TreeNode* Solution::sortedArrayToBST_workhorse(vector<int>& nums, int left, int right)
{
    if(left >= right)
    {
        return NULL;
    }
    else
    {
        int mid = left + (right - left)/2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = sortedArrayToBST_workhorse(nums, left, mid);
        root->right = sortedArrayToBST_workhorse(nums, mid+1, right);
        return root;
    }
}

vector<int> Solution::findMode(TreeNode* root)
{
    /*
        Given a binary search tree (BST) with duplicates, 
        find all the mode(s) (the most frequently occurred element) in the given BST.
    */

   vector<int> ans;

   int maxCount=0;
   int curCount=0, curVal;

    auto update = [&] (int val)
    {
        if(curCount > 0 && val == curVal)
        {
            ++curCount;
        }
        else
        {
            curCount = 1;
            curVal = val;
        }

        if(curCount > maxCount)
        {
            maxCount = curCount;
            ans.clear();
        }

        if(curCount == maxCount)
        {
            ans.push_back(val);
        }
    };

    function<void(TreeNode*)> dfs = [&] (TreeNode* node)
    {
       if(node == NULL) return;

        dfs(node->left);
        update(node->val);
       dfs(node->right);
    };

    dfs(root);

    return ans;
}

TreeNode* Solution::deleteNode(TreeNode* root, int key)
{
    // find the node to delete
    TreeNode* x = root;
    TreeNode* px = NULL;
    while(x != NULL && x->val != key)
    {
        px = x;
        if(x->val > key)
            x = x->left;
        else
            x = x->right;
    }

    if(x == NULL)  return root;

    bool rootIsDeleted =  x == root;

    auto transplant_u_with_v = [&] (TreeNode** pu, TreeNode* u, TreeNode* v)
    {
        if(*pu == NULL)
            *pu = v;
        else if(u == (*pu)->left)
            (*pu)->left = v;
        else
            (*pu)->right = v;
    };

    if(x->left == NULL)
        transplant_u_with_v(&px, x, x->right);
    else if(x->right == NULL)
        transplant_u_with_v(&px, x, x->left);
    else
    {
        TreeNode* pmxr = x;
        TreeNode* xr = x->right;
        while(xr->left != NULL)
        {
            pmxr = xr;
            xr = xr->left;
        }

        // xr->left == NULL

        if(x->right == xr)
        {
            transplant_u_with_v(&px, x, x->right);
            xr->left = x->left;
        }        
        else
        {
            transplant_u_with_v(&pmxr, xr, xr->right);
            transplant_u_with_v(&px, x, xr);
            xr->left = x->left;
            xr->right = x->right;
        }
    }

    if(0)
    {
        // for debug
        x->left = NULL;
        x->right = NULL;
        delete x;
    }

    return rootIsDeleted ? px : root;
}

TreeNode* Solution::minimum(TreeNode* root)
{
    TreeNode* y = NULL;
    while(root != NULL)
    {
        y = root;
        root = root->left;
    }
    return y;
}

void searchBST_scaffold(string input, int val, bool expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    TreeNode* actual = ss.searchBST(root, val);
    if((actual != NULL) == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(root);
}

void insertIntoBST_scaffold(string input, int val, string expectedResult)
{
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    TreeNode* actual = ss.insertIntoBST(root, val);
    if(binaryTree_equal(actual, expected))
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(actual);
    destroyBinaryTree(expected);
}

void kthSmallest_scaffold(string input, int k, int expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    int actual = ss.kthSmallest(root, k);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", " << k << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", " << k << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(root);
}

void recoverTree_scaffold(string input, string expectedResult)
{
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    ss.recoverTree(root);
    if(binaryTree_equal(root, expected))
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(root);
    destroyBinaryTree(expected);
}

void sortedArrayToBST_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    TreeNode* root = ss.sortedArrayToBST(vi);
    TreeNode* expected = stringToTreeNode(expectedResult);
    if(binaryTree_equal(root, expected))
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(root);
    destroyBinaryTree(expected);
}

void findMode_scaffold(string input, vector<int>& expectedResult)
{
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    vector<int> actual = ss.findMode(root);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input << ") failed";
    }
    destroyBinaryTree(root);
}

void deleteNode_scaffold(string input, int k, string expectedResult)
{
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    TreeNode* actual = ss.deleteNode(root, k);
    if(binaryTree_equal(actual, expected))
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << k << ", " << expectedResult << ") passed"; 
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << k << ", " << expectedResult << ") failed"; 
    }

    destroyBinaryTree(actual);
    destroyBinaryTree(expected);
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

    vector<int> expected;
    expected.push_back(2);
    findMode_scaffold("[1,null,2,2]", expected);

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
