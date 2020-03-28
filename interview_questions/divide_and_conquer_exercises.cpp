#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 169, 153, 154, 654, 315 */

struct BSTNode
{
    int val;
    int insertedTimes;
    int leftCount;
    BSTNode* left;
    BSTNode* right;

    BSTNode(int v)
    {
        val = v;
        insertedTimes = 1;
        leftCount = 0;
        left = right = NULL;
    }

    ~BSTNode()
    {
        delete left;
        delete right;
    }

    int less_and_equal() const { return insertedTimes + leftCount;}
};

int BSTNode_insert(BSTNode* root, int val)
{
    if(root->val == val)
    {
        root->insertedTimes++;
        return root->leftCount;
    }
    else if(root->val > val)
    {
        if(root->left == NULL)
        {
            root->left = new BSTNode(val);
            return 0;
        }
        else
        {
            return BSTNode_insert(root->left, val);
        }
    }
    else
    {
        if(root->right == NULL)
        {
            root->right = new BSTNode(val);
            return root->less_and_equal();
        }
        return root->less_and_equal() + BSTNode_insert(root->right, val);
    }
}

int BSTNode_insert_iterative(BSTNode* root, int val)
{
    int ans = 0;
    BSTNode* x = root;
    BSTNode* px = NULL;
    while(x != NULL)
    {
        px = x;
        if(x->val == val)
        {
            x->insertedTimes++;
            return ans + x->leftCount;
        }
        else if(x->val < val)
        {
            ans += x->less_and_equal();
            x = x->right;
        }
        else
        {
            x = x->left;
        }
    }

    BSTNode* t = new BSTNode(val);
    if(px->val > val)
    {
        px->left = t;
    }
    else
    {
        px->right = t;
    }
    return ans;
}

class Solution 
{
public:
    int majorityElement(vector<int>& nums);
    int findMin(vector<int> &num);
    TreeNode* constructMaximumBinaryTree(vector<int>& nums);
    vector<int> countSmaller(vector<int>& nums);

private:
    int majorityElement_hash(vector<int>& nums);
    int majorityElement_dac(vector<int>& nums);
    TreeNode* constructMaximumBinaryTree_workhorse(vector<int>& nums, int l, int r);
    vector<int> countSmaller_partition(vector<int>& nums);
    vector<int> countSmaller_dac(vector<int>& nums);
    vector<int> countSmaller_bst(vector<int>& nums);
};

int Solution::majorityElement(vector<int>& nums)
{
    return majorityElement_dac(nums);
}

int Solution::majorityElement_hash(vector<int>& nums)
{
    /*
        Given an array of size n, find the majority element. 
        The majority element is the element that appears more than ⌊n/2⌋ times.
    */

    unordered_map<int, int> countMap;
    const int k = nums.size() / 2;
    for(auto n: nums)
    {
        if(++countMap[n] > k)
            return n;
    }
    return -1;
}

int Solution::majorityElement_dac(vector<int>& nums)
{
    function<int(int, int)> dac = [&] (int l, int r)
    {
        if(l == r) return nums[l];
        int mid = l + (r-l)/2;
        int ml = dac(l, mid);
        int mr = dac(mid+1, r);
        if(ml == mr) return ml;
        return count(nums.begin()+l, nums.begin()+r+1, ml) 
                > count(nums.begin()+l, nums.begin()+r+1, mr) 
                ? ml 
                : mr;
    };

    return dac(0, nums.size()-1);
}

int Solution::findMin(vector<int> &nums)
{
    /*
        Assume a sorted array in ascending order has been rotated at some pivot.
        such as [1,2,3,4,5,6] becomes [4,5,6,1,2,3]. find the minimum element in
        the rotated array, your solution should run in O(logn) time in average.
    */

    function<int(int, int)> dac = [&](int l, int r)
    {
        if(l == r) return nums[l];
        int m = l + (r-l)/2;
        int ml = dac(l, m);
        int mr = dac(m+1, r);
        return std::min(ml, mr);
    };

    function<int(int, int)> dac_v2 = [&](int l, int r)
    {
        if(l == r) return nums[l];
        if(nums[l] < nums[r]) return nums[l];
        int m = l + (r-l)/2;
        return std::min(dac_v2(l, m), dac_v2(m+1, r));
    };

    return dac_v2(0, nums.size()-1);
}

TreeNode* Solution::constructMaximumBinaryTree(vector<int>& nums)
{
    /*
        Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:

        The root is the maximum number in the array.
        The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
        The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
        Construct the maximum tree by the given array and output the root node of this tree.
    */

   return constructMaximumBinaryTree_workhorse(nums, 0, nums.size());
}

TreeNode* Solution::constructMaximumBinaryTree_workhorse(vector<int>& nums, int l, int r)
{
    if(l >= r) return NULL;

    int index = l;
    int maxVal = nums[l];
    for(int i = l; i<r; ++i)
    {
        if(nums[i] > maxVal)
        {
            index = i;
            maxVal = nums[i];
        }
    }

    TreeNode* root = new TreeNode(nums[index]);
    root->left = constructMaximumBinaryTree_workhorse(nums, l, index);
    root->right = constructMaximumBinaryTree_workhorse(nums, index+1, r);
    return root;
}

vector<int> Solution::countSmaller(vector<int>& nums)
{
    /*
        You are given an integer array nums and you have to return a new counts array. 
        The counts array has the property where counts[i] is the number of smaller elements 
        to the right of nums[i].
    */

   return countSmaller_bst(nums);
}

vector<int> Solution::countSmaller_bst(vector<int>& nums)
{
    int size = nums.size();
    vector<int> ans(nums.size(), 0);
    std::unique_ptr<BSTNode> root (new BSTNode(nums[size-1]));
    for(int i=size-2; i>=0; --i)
    {
        ans[i] = BSTNode_insert_iterative(root.get(), nums[i]);
    }
    return ans;
}

vector<int> Solution::countSmaller_dac(vector<int>& nums)
{
    // Time Limit Exceed
    int size = nums.size();
    vector<int> ans(nums.size(), 0);
    function<int(int, int, int)> dac = [&](int l, int r, int key)
    {
        if(l == r) return nums[l] < key ? 1 : 0;

        int m = l + (r-l)/2;
        return dac(l, m, key) + dac(m+1, r, key);
    };

    for(int i=size-2; i>=0; --i)
    {
        ans[i] = dac(i+1, size-1, nums[i]);
    }
    return ans;
}

vector<int> Solution::countSmaller_partition(vector<int>& nums)
{
    // Time Limit Exceeded
    int size = nums.size();
    vector<int> ans(nums.size(), 0);
    for(int i=size-2; i>=0; --i)
    {
        auto it = std::partition(begin(nums)+i, end(nums), std::bind2nd(std::less<int>(), nums[i]));
        ans[i] = std::distance(begin(nums)+i, it);
    }
    return ans;
}

void majorityElement_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringToIntegerVector(input);
    int actual = ss.majorityElement(nums);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed";
    }    
}

void findMin_scaffold(string input, int expectedResult)
{
    Solution ss;
    vector<int> nums = stringToIntegerVector(input);
    int actual = ss.findMin(nums);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed";
    }    
}

void constructMaximumBinaryTree_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<int> vi = stringToIntegerVector(input);
    TreeNode* root = ss.constructMaximumBinaryTree(vi);
    TreeNode* expected = stringToTreeNode(expectedResult);
    if(binaryTree_equal(root, expected))
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }

    destroyBinaryTree(root);
    destroyBinaryTree(expected);
}

void countSmaller_scaffold(string input, string expectedResult)
{
    vector<int> vi = stringToIntegerVector(input);
    vector<int> expected = stringToIntegerVector(expectedResult);

    Solution ss;
    vector<int> actual = ss.countSmaller(vi);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running majorityElement tests:";
    TIMER_START(majorityElement);
    majorityElement_scaffold("[6,1,2,8,6,4,5,3,6,6,6,5,6,6,6,6]", 6);
    majorityElement_scaffold("[6]", 6);
    majorityElement_scaffold("[6,1,6]", 6);
    // majorityElement_scaffold("[6,1]", -1);
    TIMER_STOP(majorityElement);
    util::Log(logESSENTIAL) << "majorityElement using " << TIMER_MSEC(majorityElement) << " milliseconds\n";

    util::Log(logESSENTIAL) << "Running findMin tests:";
    TIMER_START(findMin);

    findMin_scaffold("[6]", 6);
    findMin_scaffold("[6,1]", 1);
    findMin_scaffold("[6,1,6]", 1);
    findMin_scaffold("[4,5,6,1,2,3]", 1);
    findMin_scaffold("[4,5,6,7,8,9,1,2,3]", 1);
    findMin_scaffold("[2,2,2,0,1]", 0);

    TIMER_STOP(findMin);
    util::Log(logESSENTIAL) << "findMin using " << TIMER_MSEC(findMin) << " milliseconds\n";

    util::Log(logESSENTIAL) << "Running constructMaximumBinaryTree tests:";
    TIMER_START(constructMaximumBinaryTree);
    constructMaximumBinaryTree_scaffold("[3,2,1,6,0,5]", "[6,3,5,null,2,0,null,null,1]");
    TIMER_STOP(constructMaximumBinaryTree);
    util::Log(logESSENTIAL) << "countSmaller using " << TIMER_MSEC(constructMaximumBinaryTree) << " milliseconds\n";

    util::Log(logESSENTIAL) << "Running countSmaller tests:";
    TIMER_START(countSmaller);
    countSmaller_scaffold("[5]", "[0]");
    countSmaller_scaffold("[5,5,5,5]", "[0,0,0,0]");
    countSmaller_scaffold("[5,2,6,1]", "[2,1,1,0]");
    countSmaller_scaffold("[5,2,2,6,1]", "[3,1,1,1,0]");
    countSmaller_scaffold("[1,5,2,2,6,1]", "[0,3,1,1,1,0]");
    TIMER_STOP(countSmaller);
    util::Log(logESSENTIAL) << "countSmaller using " << TIMER_MSEC(countSmaller) << " milliseconds\n";
}