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
#include <string>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <random>

#include <type_traits>
#include <typeinfo>
#include <memory>
#include <thread>

#include "util/log.h"
#include "util/timing_util.h"
#include "util/boost_assert.hpp"
#include "util/disjoint_set.h"
#include "util/dist_table_wrapper.hpp"

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int n=0):
        val(n), next(nullptr)
    {}
};

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int n=0) :
        val(n), left(nullptr), right(nullptr) {
    }

    ~TreeNode() {
        delete left;
        delete right;
    }

    bool is_leaf() { return left==nullptr && right==nullptr;}
};

void generateTestArray(std::vector<int>& input, int arraySize, bool allEqual, bool sorted=true);
void printLinkedList(ListNode* head);
void printBinaryTree(TreeNode* root);

void trimTrailingSpaces(std::string& input);
void trimLeftTrailingSpaces(std::string& input);
void trimRightTrailingSpaces(std::string& input);

ListNode* stringToListNode(std::string input);
ListNode* vectorToListNode(const std::vector<int>& vi);
TreeNode* stringToTreeNode(std::string input);

// build a height-balanced binary tree from the input array
TreeNode* vectorToTreeNode(const std::vector<int>& vi);

bool list_equal(ListNode* l1, ListNode* l2);
bool binaryTree_equal(TreeNode* t1, TreeNode* t2);

// don't call this function any more, it would result in double-free exception since
// TreeNode's destructor will try to free its children.
void destroyBinaryTree(TreeNode* root);

void destroyLinkedList(ListNode* head);

class Node {
public:
    int val;
    std::vector<Node*> neighbors;

    Node() {
        val = 0;
    }
    Node(int _val) {
        val = _val;
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

static std::vector<std::pair<int, int>> directions { // row, column
    {-1, 0}, // up
    {1, 0}, // down
    {0, -1}, // left
    {0, 1}, // right
};

struct Coordinate {
    int x, y;

    Coordinate(): x(0), y(0) {}
    Coordinate(int a, int b): x(a), y(b) {}

    bool operator<(const Coordinate& rhs) const {
        return std::tie(x, y) < std::tie(rhs.x, rhs.y);
    }

    bool operator==(const Coordinate& rhs) const {
        return std::tie(x, y) == std::tie(rhs.x, rhs.y);
    }

    Coordinate& operator+(const Coordinate& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }
};

template<class T>
std::vector<T> stringTo1DArray(std::string input) {
    std::vector<T> output;
    trimTrailingSpaces(input);
    input = input.substr(1, input.length() - 2);
    std::string item;
    char delim = ',';
    std::stringstream ss(input);
    while (std::getline(ss, item, delim)) {
        if (std::is_integral<T>::value) {
            output.push_back((T)std::stoi(item));
        } else if (std::is_floating_point<T>::value) {
            output.push_back((T)std::stod(item));
        }
    }
    return output;
}

template<class T>
std::vector<std::vector<T>> stringTo2DArray(std::string input) {
    typedef T value_type;
    std::vector<std::vector<value_type>> output;
    trimTrailingSpaces(input);
    input = input.substr(1, input.length() - 2);
    size_t pos = 0;
	while (pos < input.size()) {
		size_t last = pos;
		pos = input.find(']', pos);
		if (pos == std::string::npos) {
            break;
        }
		std::string item = input.substr(last, pos-last+1);
		output.push_back(stringTo1DArray<value_type>(item));
		pos = input.find('[', pos);
	}
    return output;
}

template<>
inline std::vector<std::string> stringTo1DArray(std::string input) {
    std::vector<std::string> output;
    trimTrailingSpaces(input);
    input = input.substr(1, input.length() - 2);
    std::string item;
    char delim = ',';
    std::stringstream ss(input);
    while (std::getline(ss, item, delim)) {
        trimTrailingSpaces(item);
        output.push_back(item);
    }
    return output;
}

template<>
inline std::vector<char> stringTo1DArray(std::string input) {
    std::vector<char> output;
    trimTrailingSpaces(input);
    input = input.substr(1, input.length() - 2);
    std::string item;
    char delim = ',';
    std::stringstream ss(input);
    while (std::getline(ss, item, delim)) {
        trimTrailingSpaces(item);
        output.push_back(item[0]);
    }
    return output;
}

// output in format like "[1,1,2]"
template<typename T>
std::string numberVectorToString(const std::vector<T>& input) {
    std::string ans;
    ans.reserve(input.size() * 4);
    ans += "[";
    for (const auto& n: input) {
        ans.append(std::to_string(n));
        ans.append(",");
    }
    if (ans.back() == ',') {
        ans.pop_back();
    }
    ans += "]";
    return ans;
}

template<>
inline std::string numberVectorToString(const std::vector<char>& input) {
    std::string ans;
    ans.reserve(input.size() * 4);
    ans += "[";
    for ( auto n: input) {
        ans.push_back(n);
        ans.push_back(',');
    }
    if (ans.back() == ',') {
        ans.pop_back();
    }
    ans += "]";
    return ans;
}

template<typename T>
void print_vector(const std::vector<T>& v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
}
