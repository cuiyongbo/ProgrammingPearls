#include "leetcode.h"

using namespace std;

class Solution {
public:
    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        vector<int> seq1, seq2;
        getLeafValueSequence(root1, seq1);
        getLeafValueSequence(root2, seq2);

        if(seq1.size() == seq2.size())
        {
            return equal(seq1.begin(), seq1.end(), seq2.begin());
        }
        else
        {
            return false;
        }
    }

private:
    void getLeafValueSequence(TreeNode* root, vector<int>& seq)
    {
        stack<TreeNode*> s;
        if(root != NULL) s.push(root);
        while(!s.empty())
        {
            auto t = s.top(); s.pop();
            if(t->left == NULL && t->right == NULL)
                seq.push_back(t->val);

            if(t->right != NULL) s.push(t->right);
            if(t->left != NULL) s.push(t->left);
        }
    }
};

void scaffold(string input1, string input2, bool expectedResult)
{
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);

    Solution ss;
    bool actual = ss.leafSimilar(t1, t2);
    cout << boolalpha << "Case: (" << input1 << ", " << input2 << ", expected<" << expectedResult << ">) ";
    cout << ((actual == expectedResult) ? "passed\n" : "failed\n");

    destroyBinaryTree(t1);
    destroyBinaryTree(t2);
}

int main()
{
    scaffold("[3,5,1,6,2,9,8,null,null,7,4]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", true);
    scaffold("[3,5,1,6,2,9,8,null,null,4,7]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", false);
}
