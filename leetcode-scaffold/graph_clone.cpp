#include "leetcode.h"

using namespace std;
using namespace osrm;

class RandomListNode {
public:
    int val;
    RandomListNode* next;
    RandomListNode* random;
    RandomListNode(int _val) {
        val = _val;
        next = nullptr;
        random = nullptr;
    }
};

/* leetcode exercise: 133, 138 */
class Solution {
public:
    Node* cloneGraph(Node* node);
    RandomListNode* copyRandomList(RandomListNode* head);
};


/*
    Given a reference of a node in a connected undirected graph. Return a deep copy (clone) of the graph.
*/
Node* Solution::cloneGraph(Node* node) {
    std::map<Node*, Node*> visited; // original node, doppelganger
    std::function<Node*(Node*)> dfs = [&] (Node* p) {
        Node* np = new Node(p->val);
        visited[p] = np;
        for (auto c: p->neighbors) {
            if (visited.count(c) != 0) {
                np->neighbors.push_back(visited[c]);
            } else {
                np->neighbors.push_back(dfs(c));
            }
        }
        return np;
    };
    return dfs(node);
}


/*
    A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
    The Linked List is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:
        val: an integer representing Node.val
        random_index: the index of the node (0-index) where random pointer points to, or null if it does not point to any node.
    Return a deep copy of the list.
*/
RandomListNode* Solution::copyRandomList(RandomListNode* head) {
    std::map<RandomListNode*, RandomListNode*> visited; // original node, doppelganger
    std::function<RandomListNode*(RandomListNode*)> dfs = [&] (RandomListNode* node) {
        if (node == nullptr) {
            return node;
        }
        if (visited.count(node) == 0) {
            visited[node] = new RandomListNode(node->val);
            visited[node]->next = dfs(node->next);
            visited[node]->random = dfs(node->random);
        }
        return visited[node];
    };
    return dfs(head);
}


void cloneGraph_scaffold(string input) {
    Solution ss;
    Node* g1 = stringToUndirectedGraph(input);
    Node* g2 = ss.cloneGraph(g1);
    if (graph_equal(g1, g2)) {
        util::Log(logINFO) << "Case(" << input << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ") failed";
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running cloneGraph tests:";
    TIMER_START(cloneGraph);
    cloneGraph_scaffold("[[1]]");
    cloneGraph_scaffold("[[2,4],[1,3],[2,4],[1,3]]");
    TIMER_STOP(cloneGraph);
    util::Log(logESSENTIAL) << "cloneGraph using " << TIMER_MSEC(cloneGraph) << " milliseconds";
}