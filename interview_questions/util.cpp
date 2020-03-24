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
