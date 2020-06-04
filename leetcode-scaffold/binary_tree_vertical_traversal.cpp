#include "leetcode.h"

using namespace std;

class Solution {
public:
    vector<vector<int>> verticalTraversal(TreeNode* root) 
    {
        vector<vector<int>> ans;
        if(root == NULL) return ans;
        
        vector<Element> elements;
        queue<Element> q;
        q.push({0, 0, root});
        while(!q.empty())
        {
            int size = q.size();
            for(int i=0; i<size; i++)
            {
                auto t = q.front(); q.pop();
                elements.push_back(t);
                if(t.node->left != NULL) q.push({t.x-1, t.y-1, t.node->left});
                if(t.node->right != NULL) q.push({t.x+1, t.y-1, t.node->right});
            }
        }
        
        sort(elements.begin(), elements.end());
        
        auto last = elements[0];
        vector<int> courier;
        for(auto e: elements)
        {
            if(last.x == e.x)
            {
                courier.push_back(e.node->val);
            }
            else
            {
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
    struct Element
    {
        int x, y;
        TreeNode* node;
        
        bool operator<(const Element& rhs) const
        {
            if(x != rhs.x)
            {
                return x < rhs.x;
            }
            else
            {
                if(y != rhs.y)
                {
                    return y > rhs.y;
                }
                else
                {
                    return node->val < rhs.node->val;
                }
            }
        }
    };
};

int main()
{
    string input = "[1,2,3,4,5,6]";
    TreeNode* root = stringToTreeNode(input);
    Solution ss;
    vector<vector<int>> ans = ss.verticalTraversal(root);
    for(auto& v: ans)
    {
        printVector(v);
    }
    return 0;
}
