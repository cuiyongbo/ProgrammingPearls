#include "leetcode.h"

using namespace std;

class Solution
{
public:
    bool isSymmetric(TreeNode* root)
    {
        return isSymmetric_iterative(root);
    }

    bool isSymmetric_recursive(TreeNode* root);
    bool isSymmetric_iterative(TreeNode* root);
};

bool Solution::isSymmetric_recursive(TreeNode* root)
{
    function<bool(TreeNode*, TreeNode*)> isMirror = [&](TreeNode* t1, TreeNode* t2)
    {
        if(t1==NULL && t2==NULL) return true;
        if(t1==NULL || t2==NULL) return false;
        return (t1->val == t2->val)
            && isMirror(t1->left, t2->right)
            && isMirror(t1->right, t2->left);
    };
    return isMirror(root, root);
}

bool Solution::isSymmetric_iterative(TreeNode* root)
{
    queue<TreeNode*> q;
    q.push(root);
    q.push(root);
    while(!q.empty())
    {
        auto t1 = q.front(); q.pop();
        auto t2 = q.front(); q.pop();
        if(t1==NULL && t2==NULL) continue;
        if(t1==NULL || t2==NULL) return false;
        if(t1->val != t2->val) return false;
        q.push(t1->left);
        q.push(t2->right);
        q.push(t1->right);
        q.push(t2->left);
    }
    return true;
}
