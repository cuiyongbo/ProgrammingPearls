#include "leetcode.h"

using namespace std;
using namespace osrm;

class BinaryTreeTextCodec
{
public:
    string serialize(TreeNode* root);
    TreeNode* deserialize(string data);

private:
    TreeNode* deserialize_workhorse(istringstream& in);
};

string BinaryTreeTextCodec::serialize(TreeNode* root)
{
    ostringstream out;
    function<void(TreeNode*)> preOrderTraversal = [&](TreeNode* node)
    {
        if(node == NULL)
        {
            out << "# ";
        }
        else
        {
            out << node->val << " ";
            preOrderTraversal(node->left);
            preOrderTraversal(node->right);
        }
    };

    preOrderTraversal(root);
    return out.str();
}

TreeNode* BinaryTreeTextCodec::deserialize(string data)
{
    istringstream in(data);
    return deserialize_workhorse(in);
}

TreeNode* BinaryTreeTextCodec::deserialize_workhorse(istringstream& in)
{
    string val;
    in >> val;
    if(val == "#")
    {
        return nullptr;
    }
    else
    {
        TreeNode* root = new TreeNode(stoi(val));
        root->left = deserialize_workhorse(in);
        root->right = deserialize_workhorse(in);
        return root;
    }
}

void textCodec_scaffold(string input, string expectedResult)
{
    BinaryTreeTextCodec codec;
    TreeNode* expected = stringToTreeNode(expectedResult);

    TreeNode* actual = codec.deserialize(input);
    if(!binaryTree_equal(actual, expected))
    {
        util::Log(logERROR) << "BinaryTreeTextCodec::deserialize failed";
        return;
    }

    string s = codec.serialize(actual);
    util::Log(logESSENTIAL) << s;
    if(s != input)
    {
        util::Log(logERROR) << "BinaryTreeTextCodec::serialize failed";
        return;
    }

    util::Log(logESSENTIAL) << "BinaryTreeTextCodec test passed";
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    textCodec_scaffold("1 2 4 # # # 3 # # ", "[1,2,3,4]");
}
