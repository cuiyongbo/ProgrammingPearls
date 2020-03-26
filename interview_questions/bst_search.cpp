#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 700, 701 */

class Solution
{
public:
    TreeNode* searchBST(TreeNode* root, int val);
    TreeNode* insertIntoBST(TreeNode* root, int val);
};

TreeNode* Solution::searchBST(TreeNode* root, int val)
{
    while(root != NULL)
    {
        if(root->val == val)
        {
            break;
        }
        else if(root->val > val)
        {
            root = root->left;
        }
        else
        {
            root = root->right;
        }
    }
    return root;
}

TreeNode* Solution::insertIntoBST(TreeNode* root, int val)
{
    TreeNode* t = new TreeNode(val);
    if(root == NULL) return t;

    TreeNode* p = root;
    TreeNode* q = NULL;
    while(p != NULL)
    {
        q = p;
        if(p->val > val)
        {
            p = p->left;
        }
        else
        {
            p = p->right;
        }
    }

    if(q->val > val)
        q->left = t;
    else
        q->right = t;

    return root;
}

void searchBST_scaffold(string input, int val, bool expectedResult)
{
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    TreeNode* actual = ss.searchBST(root, val);
    if((actual != NULL) == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(root);
}

void insertIntoBST_scaffold(string input, int val, string expectedResult)
{
    TreeNode* root = stringToTreeNode(input);
    TreeNode* expected = stringToTreeNode(expectedResult);

    Solution ss;
    TreeNode* actual = ss.insertIntoBST(root, val);
    if(binaryTree_equal(actual, expected))
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", " << val << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(actual);
    destroyBinaryTree(expected);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running searchBST tests:";
    TIMER_START(searchBST);
    searchBST_scaffold("[1,2,3]", 1, true);
    searchBST_scaffold("[4,2,7,1,3]", 1, true);
    searchBST_scaffold("[4,2,7,1,3]", 8, false);
    searchBST_scaffold("[4,2,7,1,3]", 5, false);
    searchBST_scaffold("[4,2,7,1,3]", 0, false);
    TIMER_STOP(searchBST);
    util::Log(logESSENTIAL) << "searchBST using " << TIMER_MSEC(searchBST) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running insertIntoBST tests:";
    TIMER_START(insertIntoBST);
    insertIntoBST_scaffold("[4,2,7,1,3]", 5, "[4,2,7,1,3,5]");
    insertIntoBST_scaffold("[6,2,7,1,3]", 5, "[6,2,7,1,3,null,null,null,null,null,5]");
    TIMER_STOP(insertIntoBST);
    util::Log(logESSENTIAL) << "insertIntoBST using " << TIMER_MSEC(insertIntoBST) << " milliseconds.";
}
