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

// traversal order: left -> root -> right
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

// traversal order: root -> left -> right
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

// traversal order: left -> right -> root
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

// traversal order: left -> root -> right
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
        // t->left == nullptr
        ans.push_back(t->val); // root
        p = t->right; // right
    }
    return ans;
}

// traversal order: root -> left -> right
std::vector<int> Solution::preOrderTraversal_iteration(TreeNode* root) {
    std::vector<int> ans;
    std::stack<TreeNode*> st;
    if (root != nullptr) {
        st.push(root);
    }
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        ans.push_back(t->val); // root
        // stack: last in first out
        // we need push right child first
        if (t->right != nullptr) {
            st.push(t->right);
        }
        if (t->left != nullptr) {
            st.push(t->left);
        }
    }
    return ans;    
}

// Literally, left->right->root is the inversion of root->right->left
std::vector<int> Solution::postOrderTraversal_iteration(TreeNode* root) {
    std::vector<int> ans;
    std::stack<TreeNode*> st;
    if (root != nullptr) {
        st.push(root);
    }
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        ans.push_back(t->val); // root
        // stack: last in first out
        // we need push left child first
        if (t->left != nullptr) {
            st.push(t->left);
        }
        if (t->right != nullptr) {
            st.push(t->right);
        }
    }
    // reverse [root->right->left] order to get [left->right->root] order
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

{
    struct element_t {
        int row;
        int column;
        int val;
        element_t(int r, int c, int v) {
            row = r;
            column = c;
            val = v;
        }
        // it has to be decorated with `const`
        bool operator<(const element_t& b) const {
            if (this->column < b.column) {
                return true;
            } else if (this->column == b.column) {
                return this->val < b.val;
            } else {
                return false;
            }
        }
    };
    vector<element_t> buffer;
    function<void(TreeNode*, int, int)> dfs = [&](TreeNode* node, int r, int c) {
        if (node == nullptr) {
            return;
        }
        dfs(node->left, r+1, c-1);
        buffer.emplace_back(r, c, node->val);
        dfs(node->right, r+1, c+1);
    };
    dfs(root, 0, 0);
    //SPDLOG_WARN("buffer.size={}", buffer.size());
    std::sort(buffer.begin(), buffer.end());
    vector<vector<int>> ans;
    int cur = INT32_MIN;
    vector<int> tmp;
    for (auto n: buffer) {
        if (n.column != cur) {
            cur = n.column;
            if (!tmp.empty()) {
                ans.push_back(tmp);
                tmp.clear();
            }
        }
        tmp.push_back(n.val);
    }
    if (!tmp.empty()) {
        ans.push_back(tmp);
    }
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
            SPDLOG_ERROR("Case(test_array_scale={}, array_size={}, type={}) failed.", test_array_scale, n, BinaryTreeTraversalType_toString(type));
            return;
        }
    }
    SPDLOG_INFO("Case(test_array_scale={}, type={}) passed.", test_array_scale, BinaryTreeTraversalType_toString(type));
}


void leafSimilar_scaffold(std::string input1, std::string input2, bool expectedResult) {
    TreeNode* t1 = stringToTreeNode(input1);
    TreeNode* t2 = stringToTreeNode(input2);
    std::unique_ptr<TreeNode> guard1(t1);
    std::unique_ptr<TreeNode> guard2(t2);
    Solution ss;
    bool actual = ss.leafSimilar(t1, t2);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case({}, {}, expectedResult={}) passed.", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, expectedResult={}) failed.", input1, input2, expectedResult);
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
        SPDLOG_INFO("Case({}, expectedResult={}) passed.", input1, input2);
    } else {
        SPDLOG_ERROR("Case({}, expectedResult={}) failed.", input1, input2);
        std::cout << "Actual: ";
        for (auto& t: actual) {
            std::cout << numberVectorToString(t);
        }
        std::cout << std::endl;
    }
}


int main(int argc, char* argv[]) {
    util::LogPolicy::GetInstance().Unmute();

    SPDLOG_WARN("Running leafSimilar tests:");
    TIMER_START(leafSimilar);
    leafSimilar_scaffold("[1]", "[1]", true);
    leafSimilar_scaffold("[1]", "[2]", false);
    leafSimilar_scaffold("[1,2]", "[2,2]", true);
    leafSimilar_scaffold("[1,2,3]", "[1,3,2]", false);
    leafSimilar_scaffold("[3,5,1,6,2,9,8,null,null,7,4]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", true);
    leafSimilar_scaffold("[3,5,1,6,2,9,8,null,null,4,7]", "[3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]", false);
    TIMER_STOP(leafSimilar);
    SPDLOG_WARN("leafSimilar tests using {} ms", TIMER_MSEC(leafSimilar)); 

    SPDLOG_WARN("Running verticalTraversal tests:");
    TIMER_START(verticalTraversal);
    verticalTraversal_scaffold("[3,9,20,null,null,15,7]", "[[9],[3,15],[20],[7]]");
    verticalTraversal_scaffold("[1,2,3,4,5,6,7]", "[[4],[2],[1,5,6],[3],[7]]");
    verticalTraversal_scaffold("[1,2,3,4,5,6,7,8,9]", "[[8][4][2,9][1,5,6][3][7]]");
    TIMER_STOP(verticalTraversal);
    SPDLOG_WARN("verticalTraversal tests using {} ms", TIMER_MSEC(verticalTraversal)); 

    int test_array_scale = 1000;
    if (argc > 1) {
        test_array_scale = std::atoi(argv[1]);
        if (test_array_scale <= 0) {
            cout << "test_array_scale must be positive, default to 100 if unspecified" << endl;
            return -1;
        }
    }

    SPDLOG_WARN("Running traversal tests:");
    TIMER_START(treeTraversal_test);
    treeTraversal_scaffold(test_array_scale, BinaryTreeTraversalType_preorder);
    treeTraversal_scaffold(test_array_scale, BinaryTreeTraversalType_inorder);
    treeTraversal_scaffold(test_array_scale, BinaryTreeTraversalType_postorder);
    TIMER_STOP(treeTraversal_test);
    SPDLOG_WARN("traversal tests using {} ms", TIMER_MSEC(treeTraversal_test)); 
    return 0;
}
