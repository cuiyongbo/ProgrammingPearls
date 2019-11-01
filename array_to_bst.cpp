#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <stack>

using namespace std;

struct TreeNode
{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int n)
        : val(n), left(nullptr), right(nullptr)
    {}
};

TreeNode* helper(vector<int> arr, int start, int end);

TreeNode* arrayToBst(vector<int>& arr)
{
    sort(arr.begin(), arr.end());
    return helper(arr, 0, arr.size());
}

TreeNode* helper(vector<int> arr, int start, int end)
{
    if(start >= end)
        return nullptr;

    int mid = start + (end - start)/2;
    TreeNode* root = new TreeNode(arr[mid]);
    root->left = helper(arr, start, mid);
    root->right = helper(arr, mid+1, end);
    return root;
}

vector<int> inorderTraversal(TreeNode* root)
{
    vector<int> result;
    stack<TreeNode*> s;
    TreeNode* q = root;
    while(q != nullptr || !s.empty())
    {
        while(q != nullptr)
        {
            s.push(q);
            q = q->left;
        }

        auto t = s.top();
        s.pop();
        result.push_back(t->val);
        q = t->right;
    }
    return result;
}

// do postorder traversal
void destroyBinaryTree(TreeNode* root)
{
    if(root == nullptr)
        return;

    destroyBinaryTree(root->left);
    destroyBinaryTree(root->right);
    delete root;
}

int main()
{
    vector<int> input {9, 0, 8, 7, 6, 1, 2 , 3, 4, 5};
    TreeNode* root = arrayToBst(input);
    vector<int> result = inorderTraversal(root);
    copy(result.begin(), result.end(), ostream_iterator<int>(cout, " "));
    cout << "\n";
    destroyBinaryTree(root);
    return 0;
}
