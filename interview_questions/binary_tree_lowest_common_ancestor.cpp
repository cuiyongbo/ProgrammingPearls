#include "leetcode.h"

using namespace std;

class Solution
{
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* subtreeWithAllDeepest(TreeNode* root);

private:
    TreeNode* lowestCommonAncestor_recursive(TreeNode* root, TreeNode* p, TreeNode* q);
};

TreeNode* Solution::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    return lowestCommonAncestor_recursive(root, p, q);
}

TreeNode* Solution::lowestCommonAncestor_recursive(TreeNode* root, TreeNode* p, TreeNode* q)
{
    function<TreeNode* (TreeNode*)> dfs = [&](TreeNode* node)
    {
        if(node == NULL || node == p || node == q)
            return node;
        
        TreeNode* l = dfs(node->left);
        TreeNode* r = dfs(node->right);
        if(l != NULL && r != NULL)
            return node;
        else
            return l != NULL ? l : r;
    };

    return dfs(root);
}

TreeNode* Solution::BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    int a = min(p->val, q->val);
    int b = max(p->val, q->val);
    function<TreeNode*(TreeNode*)> dfs = [&](TreeNode* t)
    {
        if(t == NULL || (a<= t->val && t->val <= b))
        {
            return t;
        }
        else if(t->val > b)
        {
            return dfs(t->left);
        }
        else
        {
            return dfs(t->right);
        }
    };

    return dfs(root);
}

TreeNode* Solution::subtreeWithAllDeepest(TreeNode* root)
{
    function<pair<int, TreeNode*>(TreeNode*)> dfs = [&] (TreeNode* t)
    {
        TreeNode* ans = NULL;
        if(t == NULL)
            return make_pair(-1, ans);

        auto l = dfs(t->left);
        auto r = dfs(t->right);

        if(l.first == r.first)
            ans = t;
        else if(l.first > r.first)
            ans = l.second;
        else
            ans = r.second;

        return make_pair(max(l.first, r.first)+1, ans);
    };

    return dfs(root).second;
}

void lowestCommonAncestor_scaffold()
{
    string input = "[1,2,3,4,5,6,7,8]";
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    TreeNode* ans;

    TreeNode* p = root->left->left->left;
    TreeNode* q = root->left->left;
    ans = ss.lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans == q) ? ") passed\n" : ") failed\n");

    q = root->left;
    ans = ss.lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans == q) ? ") passed\n" : ") failed\n");

    q = root->left->right;
    ans = ss.lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans->val == 2) ? ") passed\n" : ") failed\n");

    q = root->right->left;
    ans = ss.lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans->val == 1) ? ") passed\n" : ") failed\n");

    destroyBinaryTree(root);
}

void BST_lowestCommonAncestor_scaffold()
{
    string input = "[4,2,5,1,3,null,6]";
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    TreeNode* ans;

    TreeNode* p = root->left->left;
    TreeNode* q = root->left;
    ans = ss.BST_lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans == q) ? ") passed\n" : ") failed\n");

    q = root->left->right;
    ans = ss.BST_lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans == root->left) ? ") passed\n" : ") failed\n");

    q = root->right;
    ans = ss.BST_lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans == root) ? ") passed\n" : ") failed\n");

    q = root->right->right;
    ans = ss.BST_lowestCommonAncestor(root, p, q);
    cout << "Case: (" << p->val << ", " << q->val << ((ans == root) ? ") passed\n" : ") failed\n");

    destroyBinaryTree(root);
}


int main()
{
    lowestCommonAncestor_scaffold();
    BST_lowestCommonAncestor_scaffold();
}