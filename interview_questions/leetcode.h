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

void printVector(std::vector<int>& input);
void printLinkedList(ListNode* head);

void trimTrailingSpaces(std::string& input);
void trimLeftTrailingSpaces(std::string& input);
void trimRightTrailingSpaces(std::string& input);

std::vector<int> stringToIntegerVector(std::string input);
ListNode* stringToListNode(std::string input);
TreeNode* stringToTreeNode(std::string input);

void destroyBinaryTree(TreeNode* root);
void destroyLinkedList(ListNode* head);
