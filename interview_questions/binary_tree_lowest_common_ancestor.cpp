#include "leetcode.h"

using namespace std;

class Solution
{
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
};

TreeNode* Solution::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    if(root == NULL || root==p || root == q) return root;
    TreeNode* l = lowestCommonAncestor(root->left, p, q);
    TreeNode* r = lowestCommonAncestor(root->right, p, q);
    if(l != NULL && r != NULL) return root;
    return l != NULL ? l : r;
}

TreeNode* Solution::BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    if(p->val > q->val) swap(p, q);

    function<TreeNode*(TreeNode*)> helper = [&](TreeNode* t)
    {
        if(t==NULL || t == p || t == q)
            return t;

        if(p->val < t->val && t->val < q->val)
        {
            return t;
        }
        else if(p->val >= t->val)
        {
            return helper(t->right);
        }
        else
        {
            return helper(t->left);
        }
    };

    return helper(root);
}
