#include "leetcode.h"

using namespace std;

class Solution
{
public:
    vector<int> inOrderTraversal_recursive(TreeNode* root);
    vector<int> preOrderTraversal_recursive(TreeNode* root);
    vector<int> postOrderTraversal_recursive(TreeNode* root);
    vector<int> inOrderTraversal_iterative(TreeNode* root);
    vector<int> preOrderTraversal_iterative(TreeNode* root);
    vector<int> postOrderTraversal_iterative(TreeNode* root);
};

vector<int> Solution::inOrderTraversal_recursive(TreeNode* root) {
    vector<int> ans;
    function<void(TreeNode*)> dfs = [&](TreeNode* node) {
        if(node == NULL) return;
        dfs(node->left);
        ans.push_back(node->val);
        dfs(node->right);
    };

    dfs(root);
    return ans;
}

vector<int> Solution::preOrderTraversal_recursive(TreeNode* root) {
    vector<int> ans;
    function<void(TreeNode*)> dfs = [&](TreeNode* node) {
        if(node == NULL) return;
        ans.push_back(node->val);
        dfs(node->left);
        dfs(node->right);
    };

    dfs(root);
    return ans;
}

vector<int> Solution::postOrderTraversal_recursive(TreeNode* root) {
    vector<int> ans;
    function<void(TreeNode*)> dfs = [&](TreeNode* node) {
        if(node == NULL) return;
        dfs(node->left);
        dfs(node->right);
        ans.push_back(node->val);
    };

    dfs(root);
    return ans;
}

vector<int> Solution::inOrderTraversal_iterative(TreeNode* root) {
    vector<int> ans;
    if(root == nullptr) {
        return ans;
    }

    stack<TreeNode*> s;
    TreeNode* q = root;
    while(!s.empty() || q != nullptr) {
        while(q != nullptr) {
            s.push(q);
            q = q->left;
        }
        auto n = s.top(); s.pop();
        ans.push_back(n->val);
        q = n->right;
    }
    return ans;
}

vector<int> Solution::preOrderTraversal_iterative(TreeNode* root) {
    vector<int> ans;
    if(root == NULL) {
        return ans;
    }

    stack<TreeNode*> s; s.push(root);
    while(!s.empty()) {
        auto n = s.top(); s.pop();
        ans.push_back(n->val);

        if(n->right != NULL) {
            s.push(n->right);
        }

        if(n->left != NULL) {
            s.push(n->left);
        }
    }
    return ans;
}

vector<int> Solution::postOrderTraversal_iterative(TreeNode* root) {
    vector<int> ans;
    if (root == nullptr) {
        return ans;
    }

    stack<TreeNode*> s; s.push(root);
    while (!s.empty()) {
        auto n = s.top(); s.pop();
        ans.push_back(n->val);
        if(n->left != nullptr) {
            s.push(n->left);
        }
        if (n->right != nullptr) {
            s.push(n->right);
        }
    }
    std::reverse(ans.begin(), ans.end());
    return ans;
}

/*
1
2 3
4 5 6 7
in: 4, 2, 5, 1, 6, 3, 7
pre: 1,2,4,5,3,6,7
post: 4,5,2,6,7,3,1
*/

void preOrderTraversal_tester()
{
    string input = "[1,2,3,4,5,6,7]";
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    vector<int> expected {1,2,4,5,3,6,7};

    vector<int> actual = ss.preOrderTraversal_recursive(root);
    assert(equal(actual.begin(), actual.end(), expected.begin()));

    actual = ss.preOrderTraversal_iterative(root);
    assert(equal(actual.begin(), actual.end(), expected.begin()));
}

void inOrderTraversal_tester()
{
    string input = "[1,2,3,4,5,6,7]";
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    vector<int> expected {4, 2, 5, 1, 6, 3,7};

    vector<int> actual = ss.inOrderTraversal_recursive(root);
    assert(equal(actual.begin(), actual.end(), expected.begin()));

    actual = ss.inOrderTraversal_iterative(root);
    assert(equal(actual.begin(), actual.end(), expected.begin()));
}

void postOrderTraversal_tester()
{
    string input = "[1,2,3,4,5,6,7]";
    TreeNode* root = stringToTreeNode(input);

    Solution ss;
    vector<int> expected {4,5,2,6,7,3,1};

    vector<int> actual = ss.postOrderTraversal_recursive(root);
    assert(equal(actual.begin(), actual.end(), expected.begin()));

    actual = ss.postOrderTraversal_iterative(root);
    assert(equal(actual.begin(), actual.end(), expected.begin()));
}

int main()
{
    preOrderTraversal_tester();
    inOrderTraversal_tester();
    postOrderTraversal_tester();
}
