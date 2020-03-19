#include "leetcode.h"

using namespace std;

class Solution 
{
public:
    bool isSubtree(TreeNode* s, TreeNode* t); 
    bool isSameTree(TreeNode* node, TreeNode* t);    
};

bool Solution::isSameTree(TreeNode* node, TreeNode* t)
{
    if(node == NULL && t == NULL) 
    {
        return true;
    }
    else if(node == NULL || t == NULL)
    {
        return false;
    }
    else if(node->val != t->val)
    {
        return false;
    }
    else
    {
        return isSameTree(node->left, t->left) && isSameTree(node->right, t->right);
    }
};

bool Solution::isSubtree(TreeNode* s, TreeNode* t) 
{
    if(s == NULL && t == NULL)
    {
        return true;
    }
    else if(s == NULL || t == NULL)
    {
        return false;
    }
 
    queue<TreeNode*> q;
    q.push(s);
    while(!q.empty())
    {
        int size = q.size();
        for(int i=0; i<size; ++i)
        {
            auto sc = q.front(); q.pop();
            
            if(isSameTree(sc, t)) return true;
            
            if(sc->left != NULL) q.push(sc->left);
            if(sc->right != NULL) q.push(sc->right);
        }
    }
    return false;
}

void scaffold(string input1, string input2, bool expectedResult)
{
    TreeNode* s = stringToTreeNode(input1);
    TreeNode* t = stringToTreeNode(input2);

    Solution ss;
    bool actual = ss.isSubtree(s, t);
    if(actual == expectedResult)
    {
        cout << boolalpha << "case: (" << input1 << ", " << input2 << ", expected<" << expectedResult << ">) passed\n"; 
    }
    else
    {
        cout << boolalpha << "case: (" << input1 << ", " << input2 << ", expected<" << expectedResult << ">) failed\n"; 
    }

    destroyBinaryTree(s);
    destroyBinaryTree(t);
}

int main()
{
    scaffold("[3,4,5,1,2]", "[4,1,2]", true);
    scaffold("[4,1,2]", "[4,1,2]", true);
    scaffold("[3,4,5,1,2,null,null,null,0]", "[4,1,2]", false);
}