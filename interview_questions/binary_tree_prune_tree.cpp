#include "leetcode.h"

using namespace std;
using namespace osrm;

class Solution 
{
public:
    TreeNode* pruneTree(TreeNode* root);
    bool isSameTree(TreeNode* p, TreeNode* q);
    TreeNode* trimBST(TreeNode* root, int L, int R);
    TreeNode* removeLeafNodes(TreeNode* root, int target);
private:
    TreeNode* removeLeafNodes_helper(TreeNode* node, TreeNode* parent, int target);
};

TreeNode* Solution::pruneTree(TreeNode* root) 
{
    if(root == NULL)
    {
        return root;
    }
    else if(root->left == NULL && root->right == NULL)
    {
        return root->val == 0 ? NULL : root;
    }
    
    root->left = pruneTree(root->left);
    root->right = pruneTree(root->right);
    if(root->left == NULL && root->right == NULL)
    {
        return root->val == 0 ? NULL : root;
    }
    else
    {
        return root;
    }
}

bool Solution::isSameTree(TreeNode* p, TreeNode* q)
{
    if(p == NULL && q == NULL)
        return true;
    else if(p == NULL || q == NULL)
        return false;
    else if(p->val != q->val)
        return false;
    else
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

TreeNode* Solution::trimBST(TreeNode* root, int L, int R) 
{
    if(root == NULL)
    {
        return root;
    }
    else if(root->val < L)
    {
        return trimBST(root->right, L, R);
    }
    else if (root->val > R)
    {
        return trimBST(root->left, L, R);
    }
    else
    {
        root->left = trimBST(root->left, L, R);
        root->right = trimBST(root->right, L, R);
        return root;
    }
}

TreeNode* Solution::removeLeafNodes(TreeNode* root, int target)
{
    return removeLeafNodes_helper(root, NULL, target);
}

TreeNode* Solution::removeLeafNodes_helper(TreeNode* node, TreeNode* parent, int target)
{
    if(node == NULL) return NULL;
    
    node->left = removeLeafNodes_helper(node->left, node, target);
    node->right = removeLeafNodes_helper(node->right, node, target);
    
    if(node->left == NULL && node->right == NULL && node->val == target)
    {
        if(parent != NULL)
        {
            if(node == parent->left)
                parent->left = NULL;
            else
                parent->right = NULL;
        }
        return NULL;
    }
    else
    {
        return node;
    }
}

void prune_tree_scaffold(string input1, string input2)
{
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    TreeNode* ans = ss.pruneTree(t1);

    if (ss.isSameTree(ans, t2))
    {
        util::UnbufferedLog(logEssential) << "Case(" << input1 << ", " << input2 << "): " << "passed";
    }
    else
    {
        util::UnbufferedLog(logERROR) << "Case(" << input1 << ", " << input2 << "): " << "failed";
    }

    destroyBinaryTree(t1);
    destroyBinaryTree(t2);
}

void trim_bst_scaffold(string input1, int L, int R, string input2)
{
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    TreeNode* ans = ss.trimBST(t1, L, R);

    if (ss.isSameTree(ans, t2))
    {
        util::UnbufferedLog(logEssential) << "Case(" << input1 << ", " << L << ", " << R << ", " << input2 << "): " << "passed";
    }
    else
    {
        util::UnbufferedLog(logERROR) << "Case(" << input1 << ", " << L << ", " << R << ", " << input2 << "): " << "failed";
    }

    destroyBinaryTree(t1);
    destroyBinaryTree(t2);
}

void remove_leaf_scaffold(string input1, int target, string input2)
{
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    TreeNode* ans = ss.removeLeafNodes(t1, target);

    if (ss.isSameTree(ans, t2))
    {
        util::UnbufferedLog(logEssential) << "Case(" << input1 << ", " << target << ", " << input2 << "): " << "passed";
    }
    else
    {
        util::UnbufferedLog(logERROR) << "Case(" << input1 << ", " << target << ", " << input2 << "): " << "failed";
    }

    destroyBinaryTree(t1);
    destroyBinaryTree(t2);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    prune_tree_scaffold("[1,0,1,0,0,0,1]", "[1,null,1,null,1]");
    prune_tree_scaffold("[1,null,0,0,1]", "[1,null,0,null,1]");
    prune_tree_scaffold("[1,0,1,0,0,0,1]", "[1,null,1,null,1]");
    prune_tree_scaffold("[1,1,0,1,1,0,1,0]", "[1,1,0,1,1,null,1]");

    trim_bst_scaffold("[1,0,2]", 0, 2, "[1,0,2]");
    trim_bst_scaffold("[1,0,2]", 1, 2, "[1,null,2]");
    trim_bst_scaffold("[3,0,4,null,2,null,null,1]", 1, 3, "[3,2,null,1]");

    remove_leaf_scaffold("[1,3,3,3,2]", 3, "[1,3,null,null,2]");
    remove_leaf_scaffold("[1,2,3,2,null,2,4]", 2, "[1,null,3,null,4]");
    remove_leaf_scaffold("[1,2,null,2,null,2]", 2, "[1]");
    remove_leaf_scaffold("[1,1,1]", 1, "[]");
}
