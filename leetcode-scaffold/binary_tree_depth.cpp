#include "leetcode.h"

using namespace std;

class Solution
{
public:
    int maxDepth(TreeNode* root)
    {
        return maxDepth_iterative(root);
    }

    int minDepth(TreeNode* root);
private:
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

/*
The minimum depth is the number of nodes along the shortest path 
from the root node down to the nearest leaf node.
*/
int Solution::minDepth(TreeNode* root)
{
    int depth = 0;
    queue<TreeNode*> q;
    if(root != NULL) q.push(root);
    while(!q.empty())
    {
        ++depth;
        int size = q.size();
        for(int i=0; i<size; i++)
        {
            auto t = q.front(); q.pop();
            if(t->left == NULL && t->right == NULL)
            {
                return depth;
            }

            if(t->left != NULL) q.push(t->left);
            if(t->right != NULL) q.push(t->right);
        }
    }
    return depth;
}

void treeDepth_tester(string input, int expectedMinDepth, int expectedMaxDepth)
{
    Solution ss;
    TreeNode* root = stringToTreeNode(input);
    assert(ss.minDepth(root) == expectedMinDepth);
    assert(ss.maxDepth(root) == expectedMaxDepth);
    destroyBinaryTree(root);
}

int main()
{
    treeDepth_tester("[]", 0, 0);
    treeDepth_tester("[1]", 1, 1);
    treeDepth_tester("[1,2]", 2, 2);
    treeDepth_tester("[1,2,3,4,5]", 2, 3);
    return 0;
}
