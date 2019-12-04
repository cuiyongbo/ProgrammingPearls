#include "leetcode.h"

using namespace std;

/*
Solution: Count size of red’s subtrees

There are two situations that blue can win.
1. one of the red’s subtree has more than n>>1 nodes. Blue colorize the root of the larger subtree.
2. red and its children has size less or equal to n>>1. Blue colorize red’s parent.
*/

class Solution {
public:
    bool btreeGameWinningMove(TreeNode* root, int n, int x)
    {
        int red_left=0, red_right=0;
        function<int(TreeNode*)> nodeCount = [&](TreeNode* node)
        {
            if(node == NULL) return 0;
            int l = nodeCount(node->left);
            int r = nodeCount(node->right);
            if(node->val == x)
            {
                red_left = l;
                red_right = r;
            }
            return 1+l+r;
        };

        nodeCount(root);

        if(1+red_left+red_right <= n/2)
            return true;

        return max(red_left, red_right) > n/2;
    }
};
