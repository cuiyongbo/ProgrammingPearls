#include "leetcode.h"
#include <sstream>

using namespace std;

void generateTestArray(vector<int>& input, int arraySize, bool allEqual, bool sorted)
{
	input.resize(arraySize);
	if (allEqual)
	{
		input.assign(arraySize, rand());
	}
	else
	{
		for(int i=0; i<arraySize; ++i)
			input[i] = rand()%827396154;

		if(sorted)
			sort(input.begin(), input.end());
	}
}

void printLinkedList(ListNode* head)
{
	ListNode* p = head;
	while(p != NULL)
	{
		if(p != head)
		{
			cout << " -> ";
		}
		cout << p->val;
		p = p->next;
	}

	if(p == head)
	{
		cout << "empty list\n";
	}
	else
	{
		cout << "\n";
	}
}


void trimTrailingSpaces(std::string& input)
{
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

// in format like [1,2,3]
vector<int> stringToIntegerVector(string input)
{
	vector<int> output;
	trimTrailingSpaces(input);
	input = input.substr(1, input.length() - 2);
	string item;
	char delim = ',';
	stringstream ss(input);
	while (getline(ss, item, delim))
	{
		output.push_back(stoi(item));
	}
	return output;
}

ListNode* stringToListNode(string input)
{
	vector<int> vi = stringToIntegerVector(input);
	ListNode dummy(0);
	ListNode* p = &dummy;
	for (auto n : vi)
	{
		p->next = new ListNode(n);
		p = p->next;
	}
	return dummy.next;
}

// string in format like [1,2,null,3,null]
TreeNode* stringToTreeNode(string input)
{
	trimTrailingSpaces(input);
	input = input.substr(1, input.size() - 2);
	if (input.empty())
	{
		return nullptr;
	}

	string item;
	char delimiter = ',';
	stringstream ss(input);

	getline(ss, item, delimiter);
	TreeNode* root = new TreeNode(stoi(item));
	queue<TreeNode*> q;
	q.push(root);
	while (true)
	{
		if (!getline(ss, item, delimiter))
			break;

		auto t = q.front(); q.pop();

		// left child
		trimTrailingSpaces(item);
		if (item != "null")
		{
			t->left = new TreeNode(stoi(item));
			q.push(t->left);
		}

		if (!getline(ss, item, delimiter))
			break;

		trimTrailingSpaces(item);
		if (item != "null")
		{
			t->right = new TreeNode(stoi(item));
			q.push(t->right);
		}
	}
	return root;
}

void destroyBinaryTree(TreeNode* root)
{
	if (root == nullptr)
		return;

	// go on postorder traversal to remove all nodes
	destroyBinaryTree(root->left);
	destroyBinaryTree(root->right);
	delete root;
}

void destroyLinkedList(ListNode* head)
{
	while(head != NULL)
	{
		ListNode* p = head->next;
		delete head;
		head = p;
	}
}

bool list_equal(ListNode* l1, ListNode* l2)
{
	if(l1 == NULL && l2 == NULL)
	{
		return true;
	}
	else if(l1 == NULL || l2 == NULL)
	{
		return false;
	}

	while(l1 != NULL && l2 != NULL)
	{
		if(l1->val != l2->val)
		{
			return false;
		}

		l1 = l1->next;
		l2 = l2->next;
	}

	return l1==NULL && l2==NULL;
}

bool binaryTree_equal(TreeNode* t1, TreeNode* t2)
{
	function<bool(TreeNode*, TreeNode*)> isSame = [&](TreeNode* t1, TreeNode* t2)
	{
		if(t1 == NULL && t2 == NULL)
		{
			return true;
		}
		else if(t1 == NULL || t2 == NULL)
		{
			return false;
		}
		else if(t1->val != t2->val)
		{
			return false;
		}
		else
		{
			return isSame(t1->left, t2->left) && isSame(t1->right, t2->right);
		}
	};

	return isSame(t1, t2);
}

// in format like [[2,4],[1,3],[2,4],[1,3]]
Node* stringToUndirectedGraph(std::string& input)
{
	/*
		Test case format:

			For simplicity sake, each node's value is the same as the node's index (1-indexed). 
			For example, the first node with val = 1, the second node with val = 2, and so on. 
			The graph is represented in the test case using an adjacency list.

			Adjacency list is a collection of unordered lists used to represent a finite graph. 
			Each list describes the set of neighbors of a node in the graph.
	*/

	vector<vector<int>> adjLists = stringTo2DArray(input);

	if(adjLists.empty()) return NULL;

	int nodeCount = adjLists.size();
	vector<Node*> nodes(nodeCount, NULL);
	for(int i=0; i<nodeCount; ++i)
	{
		if(nodes[i] == NULL) 
			nodes[i] = new Node(i+1);

		for(auto n: adjLists[i])
		{
			if(nodes[n-1] == NULL)
				nodes[n-1] = new Node(n);
			
			nodes[i]->neighbors.push_back(nodes[n-1]);
		}
	}

	return nodes[0];
}

bool graph_equal(Node* g1, Node* g2)
{
	set<Node*> visited;
	function<bool(Node*, Node*)> dfs = [&](Node* g1, Node* g2)
	{
		if(g1 == NULL && g2 == NULL)
		{
			return true;
		}
		else if(g1 == NULL || g2 == NULL)
		{
			return false;
		}
		else if(g1->val != g2->val)
		{
			return false;
		}
		else
		{
			if(g1->neighbors.size() != g2->neighbors.size())
				return false;

			for(int i=0; i<g1->neighbors.size(); ++i)
			{
				if(visited.count(g1->neighbors[i]) > 0)
					continue;
				
				visited.insert(g1->neighbors[i]);
				if(!dfs(g1->neighbors[i], g2->neighbors[i]))
					return false;
			}
			return true;
		}
	};

	return dfs(g1, g2);
}

std::vector<std::vector<int>> stringTo2DArray(std::string input)
{
	trimTrailingSpaces(input);
	input = input.substr(1, input.length()-2);

	vector<vector<int>> adjLists;

	size_t pos = 0;
	while(pos < input.size())
	{
		size_t last = pos;
		pos = input.find(']', pos);
		if(pos == string::npos) break;
		string item = input.substr(last, pos-last+1);
		adjLists.push_back(stringToIntegerVector(item));
		pos = pos + 2;
	}

	return adjLists;
}

string intVectorToString(vector<int> input)
{
	string ans;
	ans += '[';
	for(auto n: input)
	{
		ans = ans + std::to_string(n) + ",";
	}
	if(ans.back() == ',') ans.pop_back();
	ans += ']';
	return ans;
}
