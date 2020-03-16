#include "leetcode.h"

using namespace std;

class Solution
{
public:
    vector<vector<int>> levelOrder(TreeNode* root);
    vector<vector<int>> levelOrder_bottomup(TreeNode* root);
    vector<int> rightSideView(TreeNode* root);
    vector<int> leftSideView(TreeNode* root);
};

vector<vector<int>> Solution::levelOrder(TreeNode* root)
{
    vector<vector<int>> ans;
    vector<int> dummy;
    queue<TreeNode*> q;
    if(root != NULL) q.push(root);
    while(!q.empty())
    {
        int size = q.size();
        dummy.clear();
        dummy.reserve(size);
        for(int i=0; i<size; i++)
        {
            auto t = q.front(); q.pop();
            dummy.push_back(t->val);
            if(t->left != NULL) q.push(t->left);
            if(t->right != NULL) q.push(t->right);
        }
        ans.push_back(dummy);
    }
    return ans;
}

vector<vector<int>> Solution::levelOrder_bottomup(TreeNode* root)
{
    deque<vector<int>> store;
    queue<TreeNode*> q;
    if(root != NULL) q.push(root);
    while(!q.empty())
    {

        int size = q.size();
        vector<int> dummy;
        dummy.reserve(size);
        for(int i=0; i<size; i++)
        {
            auto t = q.front(); q.pop();
            dummy.push_back(t->val);
            if(t->left != NULL) q.push(t->left);
            if(t->right != NULL) q.push(t->right);
        }
        store.push_front(dummy);
    }
    return vector<vector<int>> (store.begin(), store.end());
}

vector<int> Solution::rightSideView(TreeNode* root)
{
    vector<int> ans;
    queue<TreeNode*> q;
    if(root != NULL) q.push(root);
    while(!q.empty())
    {
        int size = q.size();
        for(int i=0; i<size; i++)
        {
            auto t = q.front(); q.pop();
            if(i == size-1) ans.push_back(t->val);
            if(t->left != NULL) q.push(t->left);
            if(t->right != NULL) q.push(t->right);
        }
    }
    return ans;
}

vector<int> Solution::leftSideView(TreeNode* root)
{
    vector<int> ans;
    queue<TreeNode*> q;
    if(root != NULL) q.push(root);
    while(!q.empty())
    {
        int size = q.size();
        for(int i=0; i<size; i++)
        {
            auto t = q.front(); q.pop();
            if(i == 0) ans.push_back(t->val);
            if(t->left != NULL) q.push(t->left);
            if(t->right != NULL) q.push(t->right);
        }
    }
    return ans;
}
