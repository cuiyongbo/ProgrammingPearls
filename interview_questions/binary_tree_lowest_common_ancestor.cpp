#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 235, 236, 865 */ 

class Solution
{
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* BST_lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
    TreeNode* subtreeWithAllDeepest(TreeNode* root);
};

TreeNode* Solution::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    function<TreeNode*(TreeNode*)> dfs = [&] (TreeNode* node)
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
            return t;
        else if(t->val > b)
            return dfs(t->left);
        else
            return dfs(t->right);
    };

    return dfs(root);
}

TreeNode* Solution::subtreeWithAllDeepest(TreeNode* root)
{
    /*
        Given a binary tree rooted at root, the depth of each node is the shortest distance to the root.
        A node is deepest if it has the largest depth possible among any node in the entire tree.
        The subtree of a node is that node, plus the set of all descendants of that node.
        Return the node with the largest depth such that it contains all the deepest nodes in its subtree.
    */

    function<pair<int, TreeNode*>(TreeNode*)> dfs = [&] (TreeNode* t)
    {
        TreeNode* ans = NULL;
        if(t == NULL) return make_pair(-1, ans);

        auto l = dfs(t->left);
        auto r = dfs(t->right);
        if(l.first == r.first)
            ans = t;
        else
            ans = (l.first > r.first) ? l.second : r.second;

        return make_pair(max(l.first, r.first)+1, ans);
    };

    return dfs(root).second;
}

void lowestCommonAncestor_scaffold()
{
    string input = "[1,2,3,4,5,6,7,8]";
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper (root);

    Solution ss;
    auto tester = [&](TreeNode* p, TreeNode* q, TreeNode* expected)
    {
        TreeNode* ans = ss.lowestCommonAncestor(root, p, q);
        if(ans == expected)
        {
            util::Log() << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) passed";
        }
        else
        {
            util::Log(logERROR) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) failed";
            util::Log(logERROR) << "Expected" << expected->val << ", actual: " << ans->val;
        }
    };

    tester(root->left->left->left, root->left->left, root->left->left);
    tester(root->left->left->left, root->left, root->left);
    tester(root->left->left->left, root->left->right, root->left);
    tester(root->left->left->left, root->right->left, root);
}

void BST_lowestCommonAncestor_scaffold()
{
    string input = "[4,2,5,1,3,null,6]";
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper (root);

    Solution ss;
    auto tester = [&](TreeNode* p, TreeNode* q, TreeNode* expected)
    {
        TreeNode* ans = ss.BST_lowestCommonAncestor(root, p, q);
        if(ans == expected)
        {
            util::Log() << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) passed";
        }
        else
        {
            util::Log(logERROR) << "Case: (" << p->val << ", " << q->val << ", expected<" << expected->val << ">) failed";
            util::Log(logERROR) << "Expected" << expected->val << ", actual: " << ans->val;
        }
    };

    tester(root->left->left, root->left, root->left);
    tester(root->left->left, root->left->right, root->left);
    tester(root->left->left, root->right, root);
    tester(root->left->left, root->right->right, root);
}

void subtreeWithAllDeepest_scaffold(string input, int expectedResult)
{
    TreeNode* root = stringToTreeNode(input);
    std::unique_ptr<TreeNode> doorkeeper (root);

    Solution ss;
    TreeNode* ans = ss.subtreeWithAllDeepest(root);
    if(ans->val == expectedResult)
    {
        util::Log() << "Case(" << input << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") faild";
        util::Log(logERROR) << "Expected: " << expectedResult << ", actual: " << ans->val;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running lowestCommonAncestor tests:";
    TIMER_START(lowestCommonAncestor);
    lowestCommonAncestor_scaffold();
    TIMER_STOP(lowestCommonAncestor);
    util::Log(logESSENTIAL) << "lowestCommonAncestor using " << TIMER_MSEC(lowestCommonAncestor) << " milliseconds";

    util::Log(logESSENTIAL) << "Running BST_lowestCommonAncestor tests:";
    TIMER_START(BST_lowestCommonAncestor);
    BST_lowestCommonAncestor_scaffold();
    TIMER_STOP(BST_lowestCommonAncestor);
    util::Log(logESSENTIAL) << "BST_lowestCommonAncestor using " << TIMER_MSEC(BST_lowestCommonAncestor) << " milliseconds";

    util::Log(logESSENTIAL) << "Running subtreeWithAllDeepest tests:";
    TIMER_START(subtreeWithAllDeepest);
    subtreeWithAllDeepest_scaffold("[3,5,1,6,2,0,8,null,null,7,4]", 2);
    subtreeWithAllDeepest_scaffold("[4,2,5,1,3,null,6]", 4);
    subtreeWithAllDeepest_scaffold("[1,2,3,4,5,6,7,8]", 8);
    TIMER_STOP(subtreeWithAllDeepest);
    util::Log(logESSENTIAL) << "subtreeWithAllDeepest using " << TIMER_MSEC(subtreeWithAllDeepest) << " milliseconds";
}