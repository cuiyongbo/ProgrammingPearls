#include "leetcode.h"

using namespace std;
using namespace osrm;

/*
leetcode 987
Given a binary tree, return the vertical order traversal of its nodes values.
For each node at position (X, Y), its left and right children respectively will
be at positions (X-1, Y-1) and (X+1, Y-1).
Running a vertical line from X = -infinity to X = +infinity, whenever the vertical line 
touches some nodes, we report the values of the nodes in order from top to bottom (decreasing Y coordinates).
If two nodes have the same position, then the value of the node that is reported first is the value that is smaller.
Return an list of non-empty reports in order of X coordinate.  Every report will have a list of values of nodes.
*/

class Solution {
public:
    vector<vector<int>> verticalTraversal(TreeNode* root)  {
        vector<vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }

        vector<Element> elements;
        queue<Element> q;
        q.push({0, 0, root});
        while (!q.empty()) {
            int size = q.size();
            for (int i=0; i<size; i++) {
                auto t = q.front(); q.pop();
                elements.push_back(t);
                if(t.node->left != nullptr) {
                    q.push({t.x-1, t.y-1, t.node->left});
                }
                if(t.node->right != nullptr) {
                    q.push({t.x+1, t.y-1, t.node->right});
                }
            }
        }
        
        std::sort(elements.begin(), elements.end());
        
        auto last = elements[0];
        vector<int> courier;
        for(auto e: elements) {
            if(last.x == e.x) {
                courier.push_back(e.node->val);
            } else {
                ans.push_back(courier);
                courier.clear();
                last = e;
                courier.push_back(e.node->val);
            }
        }
        ans.push_back(courier);
        return ans;
    }
private:
    struct Element {
        int x, y;
        TreeNode* node;
        
        bool operator<(const Element& rhs) const {
            if(x != rhs.x) {
                return x < rhs.x;
            } else {
                if(y != rhs.y) {
                    return y > rhs.y;
                } else {
                    return node->val < rhs.node->val;
                }
            }
        }
    };
};

void verticalTraversal_scaffold(string input1, string input2) {
    TreeNode* t1 = stringToTreeNode(input1);
    std::unique_ptr<TreeNode> guard1(t1);
    auto expected = stringTo2DArray<int>(input2);
    Solution ss;
    auto acutual = ss.verticalTraversal(t1);
    if (expected == acutual) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ") failed.";
        util::Log(logERROR) << "Actual: ";
        for (auto t: acutual) {
            util::Log(logERROR) << numberVectorToString(t);
        }
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();
    util::Log(logESSENTIAL) << "Running verticalTraversal tests:";
    TIMER_START(verticalTraversal);
    verticalTraversal_scaffold("[3,9,20,null,null,15,7]", "[[9],[3,15],[20],[7]]");
    verticalTraversal_scaffold("[1,2,3,4,5,6,7]", "[[4],[2],[1,5,6],[3],[7]]");
    TIMER_STOP(verticalTraversal);
    util::Log(logESSENTIAL) << "verticalTraversal tests using " << TIMER_MSEC(verticalTraversal) << "ms.";
    return 0;
}
