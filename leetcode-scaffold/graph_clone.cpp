#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercise: 133, 138 */

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

class Solution {
public:
	Node* cloneGraph(Node* node);
	RandomListNode* copyRandomList(RandomListNode* head);

private: 
	RandomListNode* copyRandomList_workhorse(RandomListNode* head, map<RandomListNode*, RandomListNode*>& visited);
};

Node* Solution::cloneGraph(Node* node) {
	/*
		Given a reference of a node in a connected undirected graph.
		Return a deep copy (clone) of the graph.
	*/

	map<Node*, Node*> visited;
	function<Node*(Node*)> dfs = [&] (Node* node) {
		if (node == nullptr) {
			return (Node*)nullptr;
		}
		if (visited[node] == nullptr) {
			visited[node] = new Node(node->val);
			for (auto n: node->neighbors) {
				visited[node]->neighbors.push_back(dfs(n));
			}
		}
		return visited[node];
	};
	return dfs(node);
}

RandomListNode* Solution::copyRandomList(RandomListNode* head) {
/*
	A linked list is given such that each node contains an additional 
	random pointer which could point to any node in the list or null.

	Return a deep copy of the list.

	The Linked List is represented in the input/output as a list of n nodes. 
	Each node is represented as a pair of [val, random_index] where:

	val: an integer representing Node.val
	random_index: the index of the node (0-index) where random pointer 
	points to, or null if it does not point to any node.
*/

	map<RandomListNode*, RandomListNode*> visited;
	function<RandomListNode*(RandomListNode*)> dfs = [&] (RandomListNode* node) {
		if (node == nullptr) {
			return (RandomListNode*)nullptr;
		}
		if (visited[node] == nullptr) {
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