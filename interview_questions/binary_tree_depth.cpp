#include "leetcode.h"

using namespace std;

class Solution
{
public:
    int maxDepth(TreeNode* root)
    {
        return maxDepth(root);
    }

    int maxDepth_recursive(TreeNode* root);
    int maxDepth_iterative(TreeNode* root);
};

int Solution::maxDepth_recursive(TreeNode* root)
{
    if(root==NULL) return 0;
    return max(maxDepth_recursive(root->left),
                maxDepth_recursive(root->right)) + 1;
}

int Solution::maxDepth_iterative(TreeNode* root)
{
    int depth = 0;
    queue<TreeNode*> q;
    if(root != NULL) q.push(root);
    while(!q.empty())
    {
        int size = q.size();
        for(int i=0; i<size; i++)
        {
            auto t = q.front(); q.pop();
            if(t->left != NULL) q.push(t->left);
            if(t->right != NULL) q.push(t->right);
        }
        depth++;
    }
    return depth;
}
