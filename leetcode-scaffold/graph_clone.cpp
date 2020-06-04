#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercise: 133, 138 */

class RandomListNode 
{
public:
	int val;
	RandomListNode* next;
	RandomListNode* random;

	RandomListNode() {}

	RandomListNode(int _val, RandomListNode* _next, RandomListNode* _random) {
		val = _val;
		next = _next;
		random = _random;
	}
};

class Solution 
{
public:
	Node* cloneGraph(Node* node);
	RandomListNode* copyRandomList(RandomListNode* head);

private: 
	Node* cloneGraph_workhorse(Node* node, map<Node*, Node*>& visited);
	RandomListNode* copyRandomList_workhorse(RandomListNode* head, map<RandomListNode*, RandomListNode*>& visited);
};

Node* Solution::cloneGraph(Node* node) 
{
	/*
		Given a reference of a node in a connected undirected graph.
		Return a deep copy (clone) of the graph.
	*/

	map<Node*, Node*> visited;
	return cloneGraph_workhorse(node, visited);
}

Node* Solution::cloneGraph_workhorse(Node* node, map<Node*, Node*>& visited)
{
	if(node == NULL)
	{
		return NULL;
	}
	else if(visited.count(node) > 0)
	{
		return visited[node];
	}
	else
	{
		Node* twin = new Node(node->val);
		visited[node] = twin;
		for(auto n: node->neighbors)
		{
			twin->neighbors.push_back(cloneGraph_workhorse(n, visited));
		}
		return twin;        
	}
}

RandomListNode* Solution::copyRandomList(RandomListNode* head) 
{
	RandomListNode dummy(0, NULL, NULL);
	RandomListNode* p = &dummy;
	map<RandomListNode*, RandomListNode*> visited;
	while(head != NULL)
	{
		p->next = copyRandomList_workhorse(head, visited);
		p = p->next;
		
		head = head->next;
	}
	return dummy.next;
}

RandomListNode* Solution::copyRandomList_workhorse(RandomListNode* head, map<RandomListNode*, RandomListNode*>& visited)
{
	if(head == NULL)
	{
		return NULL;
	}
	else if(visited.count(head) > 0)
	{
		return visited[head];
	}
	else
	{
		RandomListNode* node = new RandomListNode(head->val, NULL, NULL);
		visited[head] = node;
		node->random = copyRandomList_workhorse(head->random, visited);
		return node;
	}
}

void cloneGraph_scaffold(string input)
{
	Solution ss;
	Node* g1 = stringToUndirectedGraph(input);
	Node* g2 = ss.cloneGraph(g1);
	if(graph_equal(g1, g2))
	{
		util::Log(logESSENTIAL) << "Case(" << input << ") passed";
	}
	else
	{
		util::Log(logERROR) << "Case(" << input << ") failed";
	}
}

int main()
{
	util::LogPolicy::GetInstance().Unmute();

	util::Log(logESSENTIAL) << "Running cloneGraph tests:";
	TIMER_START(cloneGraph);
	cloneGraph_scaffold("[[1]]");
	cloneGraph_scaffold("[[2,4],[1,3],[2,4],[1,3]]");
	TIMER_STOP(cloneGraph);
	util::Log(logESSENTIAL) << "cloneGraph using " << TIMER_MSEC(cloneGraph) << " milliseconds";
}