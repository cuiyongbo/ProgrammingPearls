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

struct ListNode
{
	int val;
	ListNode* next;
	ListNode(int n):
		val(n), next(nullptr)
	{}
};

struct TreeNode
{
	int val;
	TreeNode* left;
	TreeNode* right;

	TreeNode(int n) :
		val(n), left(nullptr), right(nullptr)
	{}
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

std::vector<int> stringToIntegerVector(std::string input);
ListNode* stringToListNode(std::string input);
TreeNode* stringToTreeNode(std::string input);

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
