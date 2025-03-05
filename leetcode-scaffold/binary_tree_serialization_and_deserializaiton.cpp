#include "leetcode.h"

using namespace std;
using namespace osrm;

/*
leetcode 297
Serialization is the process of converting a data structure or object into a sequence 
of bits so that it can be stored in a file or memory buffer, or transmitted across a 
network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. 
There is no restriction on how your serialization/deserialization algorithm should work. 
You just need to ensure that a binary tree can be serialized to a string and this string 
can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. 
You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.
*/
class BinaryTreeTextCodec {
public:
    string serialize(TreeNode* root);
    TreeNode* deserialize(string data);
};

string BinaryTreeTextCodec::serialize(TreeNode* root) {
    ostringstream out;
    function<void(TreeNode*)> preOrderTraversal = [&] (TreeNode* node) {
        if(node == nullptr) {
            out << "# ";
        } else {
            out << node->val << " ";
            preOrderTraversal(node->left);
            preOrderTraversal(node->right);
        }
    };
    preOrderTraversal(root);
    string s = out.str();
    s.pop_back();
    return s;
}

TreeNode* BinaryTreeTextCodec::deserialize(string data) {
    istringstream in(data);
    function<TreeNode*(istringstream&)> dfs = [&] (istringstream& in) {
        string val;
        in >> val;
        if(val == "#") {
            return (TreeNode*)nullptr;
        } else {
            TreeNode* root = new TreeNode(stoi(val));
            root->left = dfs(in);
            root->right = dfs(in);
            return root;
        }
    };
    return dfs(in);
}

void BinaryTreeTextCodec_scaffold(string input, string expectedResult) {
    BinaryTreeTextCodec codec;
    TreeNode* expected = stringToTreeNode(expectedResult);
    TreeNode* actual = codec.deserialize(input);
    if(!binaryTree_equal(actual, expected)) {
        SPDLOG_ERROR("BinaryTreeTextCodec::deserialize failed");
        return;
    }
    string s = codec.serialize(actual);
    util::Log(logESSENTIAL) << s;
    if(s != input) {
        SPDLOG_ERROR("BinaryTreeTextCodec::serialize failed");
        return;
    }
    SPDLOG_INFO("BinaryTreeTextCodec tests ({}) passed", input);
}

/*
leetcode 449
Design an algorithm to serialize and deserialize a binary search tree. 
There is no restriction on how your serialization/deserialization algorithm should work. 
You just need to ensure that a binary search tree can be serialized to a string and 
this string can be deserialized to the original tree structure.

The encoded string should be as compact as possible.

Note: Do not use class member/global/static variables to store states. 
Your serialize and deserialize algorithms should be stateless.
*/

class BSTCodec {
public:
    string serialize(TreeNode* root);
    TreeNode* deserialize(string data);
private:
    TreeNode* deserialize_workhorse(string& data, int& pos, int curMin, int curMax);
};

string BSTCodec::serialize(TreeNode* root) {
    string ans;
    stack<TreeNode*> s;
    if(root != nullptr) {
        s.push(root);
    }
    // pre-order traversal
    while(!s.empty()) {
        auto t = s.top(); s.pop();
        ans.append(reinterpret_cast<const char*>(&t->val), sizeof(t->val));
        // stack: last-in, first-out
        if(t->right != nullptr) {
            s.push(t->right);
        }
        if(t->left != nullptr) {
            s.push(t->left);
        }
    }
    return ans;
}

TreeNode* BSTCodec::deserialize(string data) {
    int pos = 0;
    return deserialize_workhorse(data, pos, INT_MIN, INT_MAX);
}

TreeNode* BSTCodec::deserialize_workhorse(string& str, int& pos, int curMin, int curMax) {
    if(pos >= str.size()) {
        return nullptr;
    }
    int val = *reinterpret_cast<const int*>(str.data() + pos);
    if(val < curMin || val > curMax) {
        return nullptr;
    }
    pos += sizeof(val);
    TreeNode* root = new TreeNode(val);
    root->left = deserialize_workhorse(str, pos, curMin, val);
    root->right = deserialize_workhorse(str, pos, val, curMax);
    return root;
}

void BSTCodec_scaffold(string input) {
    TreeNode* root = stringToTreeNode(input);
    BSTCodec codec;
    string encoded_str = codec.serialize(root);
    TreeNode* decoded_tree = codec.deserialize(encoded_str);
    if(binaryTree_equal(root, decoded_tree)) {
        SPDLOG_INFO("BSTCodec test passed");
    } else {
        SPDLOG_ERROR("BSTCodec test failed");
    }
}

int main() {
    BinaryTreeTextCodec_scaffold("1 2 4 # # # 3 # #", "[1,2,3,4]");
    BSTCodec_scaffold("[5,2,6,1,4,null,8]");
    BSTCodec_scaffold("[5,2,6,null,4,null,8]");
}
