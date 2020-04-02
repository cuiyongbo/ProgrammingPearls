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

// in format like [1,2,3]
std::vector<int> stringToIntegerVector(std::string input);
std::vector<std::vector<int>> stringTo2DArray(std::string input);
ListNode* stringToListNode(std::string input);
TreeNode* stringToTreeNode(std::string input);

std::string intVectorToString(std::vector<int>& input);

bool list_equal(ListNode* l1, ListNode* l2);
bool binaryTree_equal(TreeNode* t1, TreeNode* t2);

void destroyBinaryTree(TreeNode* root);
void destroyLinkedList(ListNode* head);

class DSU
{
public:
    DSU(int count)
    {
        m_aux.resize(count);
		std::iota(m_aux.begin(), m_aux.end(), 0);
    }

    int find(int x)
    {
        if(m_aux[x] != x)
        {
            m_aux[x] = find(m_aux[x]);
        }
        return m_aux[x];
    }

    void unionFunc(int x, int y)
    {
        m_aux[find(x)] = find(y);
    }

    int groupCount()
    {
    	std::unordered_set<int> groups;
    	for(int i=0; i<m_aux.size(); ++i)
    	{
    		groups.emplace(find(i));
    	}
    	return groups.size();
    }

private:
    std::vector<int> m_aux;
};

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
