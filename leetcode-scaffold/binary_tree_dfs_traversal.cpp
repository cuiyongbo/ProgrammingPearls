#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode 94, 144, 145, 872, 987 */
class Solution {
public:
    std::vector<int> inOrderTraversal_recursion(TreeNode* root);
    std::vector<int> preOrderTraversal_recursion(TreeNode* root);
    std::vector<int> postOrderTraversal_recursion(TreeNode* root);
    std::vector<int> inOrderTraversal_iteration(TreeNode* root);
    std::vector<int> preOrderTraversal_iteration(TreeNode* root);
    std::vector<int> postOrderTraversal_iteration(TreeNode* root);
    bool leafSimilar(TreeNode* root1, TreeNode* root2);
    std::vector<std::vector<int>> verticalTraversal(TreeNode* root);
};

std::vector<int> Solution::inOrderTraversal_recursion(TreeNode* root) {
    std::vector<int> ans;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* root) {
        if (root != nullptr) {
            dfs(root->left);
            ans.push_back(root->val);
            dfs(root->right);
        }
    };
    dfs(root);
    return ans;
}

std::vector<int> Solution::preOrderTraversal_recursion(TreeNode* root) {
    std::vector<int> ans;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* root) {
        if (root != nullptr) {
            ans.push_back(root->val);
            dfs(root->left);
            dfs(root->right);
        }
    };
    dfs(root);
    return ans;
}

std::vector<int> Solution::postOrderTraversal_recursion(TreeNode* root) {
    std::vector<int> ans;
    std::function<void(TreeNode*)> dfs = [&] (TreeNode* root) {
        if (root != nullptr) {
            dfs(root->left);
            dfs(root->right);
            ans.push_back(root->val);
        }
    };
    dfs(root);
    return ans;
}

std::vector<int> Solution::inOrderTraversal_iteration(TreeNode* root) {
    std::vector<int> ans;
    std::stack<TreeNode*> st;
    TreeNode* p = root;
    while (p != nullptr || !st.empty()) {
        while (p != nullptr) {
            st.push(p);
            p = p->left; // left
        }
        auto t = st.top(); st.pop();
        ans.push_back(t->val); // root
        p = t->right; // right
    }
    return ans;
}

std::vector<int> Solution::preOrderTraversal_iteration(TreeNode* root) {
    std::vector<int> ans;
    std::stack<TreeNode*> st;
    st.push(root);
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        if (t == nullptr) {
            continue;
        }
        ans.push_back(t->val);
        st.push(t->right);
        st.push(t->left);
    }
    return ans;
}

// Literally, left->right->root is the inversion of root->right->left
std::vector<int> Solution::postOrderTraversal_iteration(TreeNode* root) {
    std::vector<int> ans;
    std::stack<TreeNode*> st;
    st.push(root);
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        if (t == nullptr) {
            continue;
        }
        ans.push_back(t->val);
        st.push(t->left);
        st.push(t->right);
    }
    std::reverse(ans.begin(), ans.end());
    return ans;
}

bool Solution::leafSimilar(TreeNode* root1, TreeNode* root2) {
/*
    Consider all the leaves of a binary tree, from left to right order, the values of leaves form a leaf value sequence.
    For example, for a tree with node [3,5,1,6,2,9,8,null,null,7,4], the leaf value sequence is (6, 7, 4, 9, 8).
    Two binary trees are considered leaf-similar if their leaf value sequence is the same. Return true if and only if the two given trees with head nodes root1 and root2 are leaf-similar.

    Hint: perform a preorder/inorder/postorder traversal to get the leaf value sequence
*/
    std::function<void(TreeNode*, std::vector<int>&)> dfs = [&] (TreeNode* node, std::vector<int>& seq) {
        if (node != nullptr) {
            if (node->left == nullptr && node->right == nullptr) {
                seq.push_back(node->val);
            }
            dfs(node->left, seq);
            dfs(node->right, seq);
        }
    };
    std::vector<int> seq1, seq2;
    dfs(root1, seq1);
    dfs(root2, seq2);
    return seq1 == seq2;
}

std::vector<std::vector<int>> Solution::verticalTraversal(TreeNode* root) {
/*
Given the root of a binary tree, calculate the vertical order traversal of the binary tree.
For each node at position (row, col), its left and right children will be at positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).
The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column.
There may be multiple nodes in the same coordinate. In such a case, sort these nodes by their values.
Return the vertical order traversal of the binary tree.
Example 1:
    Input: root = [1,2,3,4,5,6,7]
    Output: [[4],[2],[1,5,6],[3],[7]]
*/

{ // for normal binary tree
    typedef std::pair<int, int> Coordinate; //(x, y)
    std::map<Coordinate, std::vector<int>> mp; // coordinate: elements
    std::function<void(TreeNode*, Coordinate)> dfs = [&] (TreeNode* node, Coordinate coor) {
        if (node == nullptr) {
            return;
        }
        mp[coor].push_back(node->val);
        dfs(node->left, {coor.first-1, coor.second+1});
        dfs(node->right, {coor.first+1, coor.second+1});
    };
    dfs(root, {0, 0});
    int last_x = mp.begin()->first.first;
    std::vector<std::vector<int>> ans;
    std::vector<int> buffer;
    for (auto& it: mp) {
        std::sort(it.second.begin(), it.second.end());
        if (it.first.first != last_x) {
            last_x = it.first.first;
            ans.push_back(buffer);
            buffer.clear();
        }
        buffer.insert(buffer.end(), it.second.begin(), it.second.end());
    }
    ans.push_back(buffer);
    return ans;
}


{ // if root is a binary search tree    
    typedef std::pair<int, int> Coordinate;
    std::map<int, std::vector<int>> mp;
    std::function<void(TreeNode*, Coordinate)> dfs = [&] (TreeNode* node, Coordinate coor) { // inorder traversal
        if (node != nullptr) {
            dfs(node->left, std::make_pair(coor.first-1, coor.second-1));
            mp[coor.first].push_back(node->val);
            dfs(node->right, std::make_pair(coor.first+1, coor.second-1));
        }
    };
    dfs(root, std::make_pair(0, 0));
    std::vector<std::vector<int>> ans;
    for (auto& p : mp) {
        ans.push_back(p.second);
    }
    return ans;
}

}

enum BinaryTreeTraversalType {
    BinaryTreeTraversalType_preorder, // root, left, right
    BinaryTreeTraversalType_inorder, // left, root, right
    BinaryTreeTraversalType_postorder, // left, right, root
};

const char* BinaryTreeTraversalType_toString(BinaryTreeTraversalType type) {
    const char* str = nullptr;
    switch (type) {
    case BinaryTreeTraversalType::BinaryTreeTraversalType_preorder:
        str = "preorder";
        break;
    case BinaryTreeTraversalType::BinaryTreeTraversalType_inorder:
        str = "inorder";
        break;
    case BinaryTreeTraversalType::BinaryTreeTraversalType_postorder:
        str = "postorder";
        break;
    default:
        str = "unknown";
        break;
    }
    return str;
}

void treeTraversal_scaffold(int test_array_scale, BinaryTreeTraversalType type) {
    Solution ss;
    //using traversal_funct_t = std::vector<int> (Solution::*) (TreeNode*);
    typedef std::vector<int> (Solution::*traversal_funct_t) (TreeNode*);

    std::pair<traversal_funct_t, traversal_funct_t> couples[] = {
        {&Solution::preOrderTraversal_iteration, &Solution::preOrderTraversal_recursion},
        {&Solution::inOrderTraversal_iteration, &Solution::inOrderTraversal_iteration},
        {&Solution::postOrderTraversal_iteration, &Solution::postOrderTraversal_recursion},
    };

    std::vector<int> vi; vi.reserve(test_array_scale);
    for (int i=0; i<1000; ++i) {
        vi.clear();
        int n = rand() % test_array_scale;
        for (int j=0; j<n; ++j) {
            vi.push_back(rand());
        }

        TreeNode* root = vectorToTreeNode(vi);
        auto ans1 = std::invoke(couples[type].first, ss, root); // have to use c++17
        auto ans2 = std::invoke(couples[type].second, ss, root);
        if(ans1 != ans2) {
            util::Log(logERROR) << "Case(test_array_scale<" << test_array_scale 
                << ">, array_size<" << n << ">, "  << BinaryTreeTraversalType_toString(type) << ") failed";
        }
    }
}

void leafSimilar_scaffold(std::string input1, std::string input2, bool expectedResult) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);

    Solution ss;
    bool actual = ss.leafSimilar(t1, t2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") failed.";
    }
}

void verticalTraversal_scaffold(std::string input1, std::string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    std::unique_ptr<TreeNode> guard1(t1);
    auto expected = stringTo2DArray<int>(input2);
    Solution ss;
    auto actual = ss.verticalTraversal(t1);
    for (auto& p: actual) {
        std::sort(p.begin(), p.end());
    }
    for (auto& p: expected) {
        std::sort(p.begin(), p.end());
    }
    if (expected == actual) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed. Actual: ";
        for (auto& t: actual) {
            util::Log(logERROR) << numberVectorToString(t);
        }
    }
}

int main(int argc, char* argv[]) {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running leafSimilar tests:";
    TIMER_START(leafSimilar);
    leafSimilar_scaffold("[1]", "[1]", true);
    leafSimilar_scaffold("[1]", "[2]", false);
    leafSimilar_scaffold("[1,2]", "[2,2]", true);
    leafSimilar_scaffold("[1,2,3]", "[1,3,2]", false);
    leafSimilar_scaffold("[3,5,1,6,2,9,8,null,null,7,4]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", true);
    leafSimilar_scaffold("[3,5,1,6,2,9,8,null,null,4,7]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", false);
    TIMER_STOP(leafSimilar);
    util::Log(logESSENTIAL) << "Running leafSimilar tests uses " << TIMER_MSEC(leafSimilar) << "ms.";

    util::Log(logESSENTIAL) << "Running verticalTraversal tests:";
    TIMER_START(verticalTraversal);
    verticalTraversal_scaffold("[3,9,20,null,null,15,7]", "[[9],[3,15],[20],[7]]");
    verticalTraversal_scaffold("[1,2,3,4,5,6,7]", "[[4],[2],[1,5,6],[3],[7]]");
    TIMER_STOP(verticalTraversal);
    util::Log(logESSENTIAL) << "verticalTraversal tests using " << TIMER_MSEC(verticalTraversal) << "ms.";

    int test_array_scale = 100;
    if (argc > 1) {
        test_array_scale = std::atoi(argv[1]);
        if (test_array_scale <= 0) {
            cout << "test_array_scale must be positive, default to 100 if unspecified" << endl;
            return -1;
        }
    }

    util::Log(logESSENTIAL) << "Running traversal tests:";
    TIMER_START(treeTraversal_test);
    treeTraversal_scaffold(test_array_scale, BinaryTreeTraversalType_preorder);
    treeTraversal_scaffold(test_array_scale, BinaryTreeTraversalType_inorder);
    treeTraversal_scaffold(test_array_scale, BinaryTreeTraversalType_postorder);
    TIMER_STOP(treeTraversal_test);
    util::Log(logESSENTIAL) << "traversal tests using " << TIMER_MSEC(treeTraversal_test) << " milliseconds";
    
    return 0;
}
