#include "leetcode.h"
#include <sstream>

using namespace std;

void generateTestArray(vector<int>& input, int arraySize, bool allEqual, bool sorted) {
    input.resize(arraySize);
    if (allEqual) {
        input.assign(arraySize, rand());
    } else {
        for (int i=0; i<arraySize; ++i) {
            input[i] = rand()%827396154;
        }
        if (sorted) {
            sort(input.begin(), input.end());
        }
    }
}

void printLinkedList(ListNode* head) {
    ListNode* p = head;
    while (p != nullptr) {
        if (p != head) {
            cout << " -> ";
        }
        cout << p->val;
        p = p->next;
    }

    if (p == head) {
        cout << "empty list\n";
    } else {
        cout << "\n";
    }
}

void printBinaryTree(TreeNode* root) {
    std::queue<TreeNode*> q;
    q.push(root);
    stringstream ss;
    ss << "tree by level: ";
    while (!q.empty()) {
        int sz = q.size();
        std::string level("[");
        for (int i=0; i<sz; ++i) {
            auto t = q.front(); q.pop();
            if (t == nullptr) {
                level.append("null");
            } else {
                level.append(std::to_string(t->val));
                if (t->left != nullptr || t->right != nullptr) {
                    q.push(t->left);
                    q.push(t->right);
                }
            }
            level.push_back(',');
        }
        level.push_back(']');
        ss << level;
    }
    cout << ss.str() << endl;
}

void trimTrailingSpaces(std::string& input) {
    trimLeftTrailingSpaces(input);
    trimRightTrailingSpaces(input);
}

void trimLeftTrailingSpaces(string& input)
{
    input.erase(input.begin(), find_if(input.begin(), input.end(), [](int ch) {
        return !isspace(ch);}));
}

void trimRightTrailingSpaces(string& input)
{
    input.erase(find_if(input.rbegin(), input.rend(), [](int ch) {
        return !isspace(ch);
        }).base(), input.end());
}

ListNode* stringToListNode(string input) {
    vector<int> vi = stringTo1DArray<int>(input);
    return vectorToListNode(vi);
}

ListNode* vectorToListNode(const std::vector<int>& vi) {
    ListNode dummy(0);
    ListNode* p = &dummy;
    for (auto n : vi) {
        p->next = new ListNode(n);
        p = p->next;
    }
    return dummy.next;
}

// string in format like [1,2,null,3,null]
TreeNode* stringToTreeNode(string input) {
    trimTrailingSpaces(input);
    input = input.substr(1, input.size() - 2);
    if (input.empty()) {
        return nullptr;
    }

    string item;
    char delimiter = ',';
    stringstream ss(input);

    getline(ss, item, delimiter);
    TreeNode* root = new TreeNode(stoi(item));
    queue<TreeNode*> q;
    q.push(root);
    while (true) {
        auto t = q.front(); q.pop();

        // left child
        if (!getline(ss, item, delimiter)) {
            break;
        }
        trimTrailingSpaces(item);
        if (item != "null") {
            t->left = new TreeNode(stoi(item));
            q.push(t->left);
        }

        // right child
        if (!getline(ss, item, delimiter)) {
            break;
        }
        trimTrailingSpaces(item);
        if (item != "null") {
            t->right = new TreeNode(stoi(item));
            q.push(t->right);
        }
    }
    return root;
}

TreeNode* vectorToTreeNode(const std::vector<int>& vi) {
    function<TreeNode*(int, int)> dfs = [&] (int l, int r) {
        if (l > r) {
            return (TreeNode*)nullptr;
        }
        int m = l + (r-l)/2;
        TreeNode* node = new TreeNode(vi[m]);
        node->left = dfs(l, m-1);
        node->right = dfs(m+1, r);
        return node;
    };
    return dfs(0, vi.size()-1);
}

void destroyBinaryTree(TreeNode* root) {
    if (root == nullptr) {
        return;
    }

    // perform a postorder traversal to remove all nodes
    destroyBinaryTree(root->left);
    destroyBinaryTree(root->right);
    delete root;
}

void destroyLinkedList(ListNode* head) {
    while (head != nullptr) {
        ListNode* p = head->next;
        delete head;
        head = p;
    }
}

bool list_equal(ListNode* l1, ListNode* l2) {
    if (l1 == nullptr && l2 == nullptr) {
        return true;
    } else if (l1 == nullptr || l2 == nullptr) {
        return false;
    }

    while (l1 != nullptr && l2 != nullptr) {
        if (l1->val != l2->val) {
            return false;
        }

        l1 = l1->next;
        l2 = l2->next;
    }

    return l1 == nullptr && l2 == nullptr;
}

bool binaryTree_equal(TreeNode* t1, TreeNode* t2) {
    if (t1 == nullptr && t2 == nullptr) {
        return true;
    } else if (t1 == nullptr || t2 == nullptr) {
        return false;
    } else if (t1->val != t2->val) {
        return false;
    } else {
        return binaryTree_equal(t1->left, t2->left) && binaryTree_equal(t1->right, t2->right);
    }
}

// in format like [[2,4],[1,3],[2,4],[1,3]]
Node* stringToUndirectedGraph(std::string& input) {
/*
    Test case format:
        For sake of simplicity, each node's value is the same as the node's index (1-indexed).
        For example, the first node with val = 1, the second node with val = 2, and so on.
        The graph is represented in the test case using an adjacency list.
        Adjacency list is a collection of unordered lists used to represent a finite graph.
        Each list describes the set of neighbors of a node in the graph.
*/

    vector<vector<int>> adjLists = stringTo2DArray<int>(input);
    if (adjLists.empty()) {
        return nullptr;
    }
    int nodeCount = adjLists.size();
    vector<Node*> nodes(nodeCount, nullptr);
    for (int i=0; i<nodeCount; ++i) {
        if (nodes[i] == nullptr) {
            nodes[i] = new Node(i+1);
        }
        for (auto n: adjLists[i]) {
            if (nodes[n-1] == nullptr) {
                nodes[n-1] = new Node(n);
            }
            nodes[i]->neighbors.push_back(nodes[n-1]);
        }
    }
    return nodes[0];
}

bool graph_equal(Node* g1, Node* g2) {
    set<Node*> visited;
    function<bool(Node*, Node*)> dfs = [&](Node* g1, Node* g2) {
        if (g1 == nullptr && g2 == nullptr) {
            return true;
        } else if (g1 == nullptr || g2 == nullptr) {
            return false;
        } else if (g1->val != g2->val) {
            return false;
        } else {
            int sz = g1->neighbors.size();
            if (sz != (int)g2->neighbors.size()) {
                return false;
            }
            for (int i=0; i<sz; ++i) {
                if (visited.count(g1->neighbors[i]) == 0) {
                    visited.insert(g1->neighbors[i]);
                    if (!dfs(g1->neighbors[i], g2->neighbors[i])) {
                        return false;
                    }
                }
            }
            return true;
        }
    };
    return dfs(g1, g2);
}
