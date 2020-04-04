#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <functional>
#include <cstring>
#include <cassert>

#include "log.h"
#include "timing_util.h"
#include "disjoint_set.h"

struct ListNode
{
	int val;
	ListNode* next;
	ListNode(int n=0):
		val(n), next(nullptr)
	{}
};

struct TreeNode
{
	int val;
	TreeNode* left;
	TreeNode* right;

	TreeNode(int n=0) :
		val(n), left(nullptr), right(nullptr)
	{}

    ~TreeNode()
    {
        delete left;
        delete right;
    }
};

template<class T>
void printVector(std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
}

void generateTestArray(std::vector<int>& input, int arraySize, bool allEqual, bool sorted=true);
void printLinkedList(ListNode* head);

void trimTrailingSpaces(std::string& input);
void trimLeftTrailingSpaces(std::string& input);
void trimRightTrailingSpaces(std::string& input);

// output in format like "[1,1,2]"
template<typename T>
std::string numberVectorToString(std::vector<T>& input)
{
	std::string ans;
	ans.reserve(input.size() * 4);
	ans += "[";
	for(const auto& n: input)
	{
		ans.append(std::to_string(n));
		ans.append(",");
	}
	if(ans.back() == ',') ans.pop_back();
	ans += "]";
	return ans;
}

// in format like [1.0, 2.0]
std::vector<double> stringToDoubleVector(std::string input);

// in format like [1,2,3]
std::vector<int> stringToIntegerVector(std::string input);
std::vector<std::vector<int>> stringTo2DArray(std::string input);

ListNode* stringToListNode(std::string input);
TreeNode* stringToTreeNode(std::string input);

// in format like [a, b]
std::vector<std::string> toStringArray(std::string input);
std::vector<std::vector<std::string>> to2DStringArray(std::string input);

bool list_equal(ListNode* l1, ListNode* l2);
bool binaryTree_equal(TreeNode* t1, TreeNode* t2);

void destroyBinaryTree(TreeNode* root);
void destroyLinkedList(ListNode* head);

class Node
{
public:
    int val;
    std::vector<Node*> neighbors;

    Node() {
        val = 0;
        neighbors = std::vector<Node*>();
    }

    Node(int _val) {
        val = _val;
        neighbors = std::vector<Node*>();
    }

    Node(int _val, std::vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

/*
    Format like [[2,4],[1,3],[2,4],[1,3]].
    For simplicity sake, each node's value is the same as the node's index (1-indexed).
    For example, the first node with val = 1, the second node with val = 2, and so on.
    The graph is represented in the test case using an adjacency list.

    Adjacency list is a collection of unordered lists used to represent a finite graph.
    Each list describes the set of neighbors of a node in the graph.
*/
Node* stringToUndirectedGraph(std::string& input);

bool graph_equal(Node* g1, Node* g2);

// up, down, left, right
static std::vector<std::vector<int>> DIRECTIONS {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

struct Coor
{
    int x, y;

    Coor(): x(0), y(0) {}
    Coor(int a, int b): x(a), y(b) {}

    bool operator<(const Coor& rhs) const
    {
        return std::tie(x, y) < std::tie(rhs.x, rhs.y);
    }
};
