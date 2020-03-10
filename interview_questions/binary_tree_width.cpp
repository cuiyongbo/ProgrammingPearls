#include "leetcode.h"

using namespace std;

class Solution
{
public:
    int widthOfBinaryTree(TreeNode* root)
    {
        int width = 0;
        queue<pair<TreeNode*, int>> q;
        if(root != NULL)
        {
            q.push(make_pair(root, 0));
        }

        while(!q.empty())
        {
            int start = 0;
            int left=0, right=0;
            int size = q.size();
            for(int i=0; i<size; ++i)
            {
                auto p = q.front(); q.pop();
                if(i==0)
                {
                    left = p.second;
                    start = left*2;
                }

                if(i == size-1)
                {
                    right = p.second;
                }

                if(p.first->left != NULL)
                    q.push(make_pair(p.first->left, p.second*2 - start));

                if(p.first->right != NULL)
                    q.push(make_pair(p.first->right, p.second*2+1 - start);
            }

            width = max(width, right-left+1);
        }
        return width;
    }
};
