#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 297, 449 */
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

void BinaryTreeTextCodec_scaffold(string input, string expectedResult)
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
    // util::Log(logESSENTIAL) << s;
    if(s != input)
    {
        util::Log(logERROR) << "BinaryTreeTextCodec::serialize failed";
        return;
    }

    util::Log(logESSENTIAL) << "BinaryTreeTextCodec test passed";
}

class BSTCodec
{
public:
    string serialize(TreeNode* root);
    TreeNode* deserialize(string data);
private:
    TreeNode* deserialize_workhorse(string& data, int& pos, int curMin, int curMax);
};

string BSTCodec::serialize(TreeNode* root)
{
    string ans;
    stack<TreeNode*> s;
    if(root != NULL) s.push(root);
    while(!s.empty())
    {
        auto t = s.top(); s.pop();
        if(t->right != NULL) s.push(t->right);
        if(t->left != NULL) s.push(t->left);
        ans.append(reinterpret_cast<const char*>(&t->val), sizeof(t->val));
        // util::Log(logESSENTIAL) << t->val;
    }
    return ans;
}

TreeNode* BSTCodec::deserialize(string data)
{
    int pos = 0;
    return deserialize_workhorse(data, pos, INT_MIN, INT_MAX);
}

TreeNode* BSTCodec::deserialize_workhorse(string& str, int& pos, int curMin, int curMax)
{
    if(pos >= str.size()) return NULL;
    int val = *reinterpret_cast<const int*>(str.data() + pos);
    if(val < curMin || val > curMax) 
        return NULL;

    pos += sizeof(val);
    TreeNode* root = new TreeNode(val);
    root->left = deserialize_workhorse(str, pos, curMin, val);
    root->right = deserialize_workhorse(str, pos, val, curMax);
    return root;
}

void BSTCodec_scaffold(string input)
{
    TreeNode* root = stringToTreeNode(input);
    
    BSTCodec codec;
    string encoded_str = codec.serialize(root);
    TreeNode* decoded_tree = codec.deserialize(encoded_str);
    if(binaryTree_equal(root, decoded_tree))
    {
        util::Log(logESSENTIAL) << "BSTCodec test passed";
    }
    else
    {
        util::Log(logERROR) << "BSTCodec test failed";
    }
    
    destroyBinaryTree(root);
    destroyBinaryTree(decoded_tree);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    BinaryTreeTextCodec_scaffold("1 2 4 # # # 3 # # ", "[1,2,3,4]");
    BSTCodec_scaffold("[5,2,6,1,4,null,8]");
}
