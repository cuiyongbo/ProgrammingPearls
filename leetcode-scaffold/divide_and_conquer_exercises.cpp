#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 169, 153, 154, 654, 315, 912 */

struct BSTNode {
    int val;
    int insertedTimes;
    int leftCount;
    BSTNode* left;
    BSTNode* right;

    BSTNode(int v) 
        : val(v), insertedTimes(1), leftCount(0),
            left(nullptr), right(nullptr) {
    }

    ~BSTNode() {
        delete left;
        delete right;
    }

    int less_and_equal() const { 
        return insertedTimes + leftCount;
    }
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
        if(root->left == nullptr)
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
        if(root->right == nullptr)
        {
            root->right = new BSTNode(val);
            return root->less_and_equal();
        }
        return root->less_and_equal() + BSTNode_insert(root->right, val);
    }
}

int BSTNode_insert_iterative(BSTNode* root, int val) {
    int ans = 0;
    BSTNode* x = root;
    BSTNode* px = nullptr;
    while (x != nullptr) {
        px = x;
        if(x->val == val) {
            x->insertedTimes++;
            return ans + x->leftCount;
        } else if(x->val < val) {
            ans += x->less_and_equal();
            x = x->right;
        } else {
            x = x->left;
        }
    }

    BSTNode* t = new BSTNode(val);
    if(px->val > val) {
        px->left = t;
    } else {
        px->right = t;
    }
    return ans;
}

class Solution {
public:
    int majorityElement(vector<int>& nums);
    int findMin(vector<int> &num);
    TreeNode* constructMaximumBinaryTree(vector<int>& nums);
    vector<int> countSmaller(vector<int>& nums);
    vector<int> sortArray(vector<int>& nums);

private:
    int majorityElement_hash(vector<int>& nums);
    int majorityElement_dac(vector<int>& nums);
    TreeNode* constructMaximumBinaryTree_workhorse(vector<int>& nums, int l, int r);
    vector<int> countSmaller_dac(vector<int>& nums);
    vector<int> countSmaller_bst(vector<int>& nums);
};

vector<int> Solution::sortArray(vector<int>& nums) {
    auto partitioner = [&] (int l, int r) {
        int p = nums[r-1];
        int i=l-1; 
        for (int j=l; j<r; j++) {
            if (nums[j] < p) {
                std::swap(nums[++i], nums[j]);
            }
        }
        std::swap(nums[i+1], nums[r-1]);
        return i+1;
    };

    function<void(int, int)> quickSort_iterative = [&](int l, int r) {
        stack<std::pair<int, int>> s;
        if (l < r) {
            s.push({l, r});
        }
        while (!s.empty()) {
            auto range = s.top(); s.pop();
            int m = partitioner(range.first, range.second);
            if (m+1 < range.second) {
                s.push({m+1, range.second});
            }
            if (range.first < m) {
                s.push({range.first, m});
            }
        }
    };

    //quickSort_iterative(0, nums.size());

    function<void(int, int)> quickSort_recursive = [&](int l, int r) {
        if (l >= r) {
            return;
        }
        int m = partitioner(l ,r);
        quickSort_recursive(l, m);
        quickSort_recursive(m+1, r);
    };

    quickSort_recursive(0, nums.size());

    return nums;
}

int Solution::majorityElement(vector<int>& nums) {
/*
    Given an array of size n, find the majority element. 
    The majority element is the element that appears more than ⌊ n/2 ⌋ times.
    You may assume that the array is non-empty and the majority element always exist in the array.
*/

    return majorityElement_dac(nums);
}

int Solution::majorityElement_hash(vector<int>& nums) {
    unordered_map<int, int> countMap;
    const int k = nums.size() / 2;
    for (auto n: nums) {
        if(++countMap[n] > k) {
            return n;
        }
    }
    return -1;
}

int Solution::majorityElement_dac(vector<int>& nums)
{
    function<int(int, int)> dac = [&] (int l, int r) {
        if(l == r) {
            return nums[l];
        }
        int mid = l + (r-l)/2;
        int ml = dac(l, mid);
        int mr = dac(mid+1, r);
        if(ml == mr) {
            return ml;
        }
        return count(nums.begin()+l, nums.begin()+r+1, ml) 
                > count(nums.begin()+l, nums.begin()+r+1, mr) 
                ? ml 
                : mr;
    };

    return dac(0, nums.size()-1);
}

int Solution::findMin(vector<int> &nums) {
    /*
        Assume a sorted array in ascending order has been rotated at some pivot.
        such as [1,2,3,4,5,6] becomes [4,5,6,1,2,3]. find the minimum element in
        the rotated array, your solution should run in O(logn) time in average.
    */

    function<int(int, int)> dac = [&](int l, int r) {
        if(l == r) {
            return nums[l];
        }
        if(nums[l] < nums[r]) {
            return nums[l];
        }
        int m = l + (r-l)/2;
        return std::min(dac(l, m), dac(m+1, r));
    };
    return dac(0, nums.size()-1);
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
    if(l >= r) return nullptr;

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

vector<int> Solution::countSmaller(vector<int>& nums) {
    /*
        You are given an integer array nums and you have to return a new counts array. 
        The counts array has the property where counts[i] is the number of smaller elements 
        to the right of nums[i].
    */

/*
    // brute force
    int size = nums.size();
    vector<int> ans(nums.size(), 0); 
    for (int i=size-1; i>=0; i--) {
        for (int j=i+1; i<size; j++) {
            if (nums[i] > nums[j]) {
                ans[i]++;
            }
        }
    } 
    return ans;
*/
   return countSmaller_bst(nums);
}

vector<int> Solution::countSmaller_bst(vector<int>& nums)
{
    int size = nums.size();
    vector<int> ans(nums.size(), 0);
    std::unique_ptr<BSTNode> root (new BSTNode(nums[size-1]));
    for(int i=size-2; i>=0; --i) {
        ans[i] = BSTNode_insert_iterative(root.get(), nums[i]);
    }
    return ans;
}

vector<int> Solution::countSmaller_dac(vector<int>& nums) {
    // Time Limit Exceed
    int size = nums.size();
    vector<int> ans(nums.size(), 0);
    function<int(int, int, int)> dac = [&](int l, int r, int key) {
        if(l == r) {
            return nums[l] < key ? 1 : 0;
        }

        int m = l + (r-l)/2;
        return dac(l, m, key) + dac(m+1, r, key);
    };

    for(int i=size-2; i>=0; --i) {
        ans[i] = dac(i+1, size-1, nums[i]);
    }
    return ans;
}

void majorityElement_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.majorityElement(nums);
    if (actual == expectedResult) {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed";
    }    
}

void findMin_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> nums = stringTo1DArray<int>(input);
    int actual = ss.findMin(nums);
    if (actual == expectedResult) {
        util::Log(logESSENTIAL) << "Case(" << input << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", " << expectedResult << ") failed";
    }    
}

void constructMaximumBinaryTree_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<int> vi = stringTo1DArray<int>(input);
    TreeNode* root = ss.constructMaximumBinaryTree(vi);
    TreeNode* expected = stringToTreeNode(expectedResult);
    if(binaryTree_equal(root, expected)) {
        util::Log(logESSENTIAL) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed";
    }
}

void countSmaller_scaffold(string input, string expectedResult) {
    vector<int> vi = stringTo1DArray<int>(input);
    vector<int> expected = stringTo1DArray<int>(expectedResult);

    Solution ss;
    vector<int> actual = ss.countSmaller(vi);
    if(actual == expected) {
        util::Log(logINFO) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed，acutal: " << numberVectorToString(actual);
    }
}

void sortArray_scaffold(string input, string expectedResult) {
    vector<int> vi = stringTo1DArray<int>(input);
    vector<int> expected = stringTo1DArray<int>(expectedResult);

    Solution ss;
    vector<int> actual = ss.sortArray(vi);
    if(actual == expected) {
        util::Log(logINFO) << "Case (" << input  << ", expected: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case (" << input  << ", expected: " << expectedResult << ") failed, actual: " << numberVectorToString(actual);
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
    util::Log(logESSENTIAL) << "majorityElement using " << TIMER_MSEC(majorityElement) << " milliseconds";

    util::Log(logESSENTIAL) << "Running findMin tests:";
    TIMER_START(findMin);
    findMin_scaffold("[6]", 6);
    findMin_scaffold("[6,1]", 1);
    findMin_scaffold("[6,1,6]", 1);
    findMin_scaffold("[4,5,6,1,2,3]", 1);
    findMin_scaffold("[4,5,6,7,8,9,1,2,3]", 1);
    findMin_scaffold("[2,2,2,0,1]", 0);
    TIMER_STOP(findMin);
    util::Log(logESSENTIAL) << "findMin using " << TIMER_MSEC(findMin) << " milliseconds";

    util::Log(logESSENTIAL) << "Running constructMaximumBinaryTree tests:";
    TIMER_START(constructMaximumBinaryTree);
    constructMaximumBinaryTree_scaffold("[3,2,1,6,0,5]", "[6,3,5,null,2,0,null,null,1]");
    TIMER_STOP(constructMaximumBinaryTree);
    util::Log(logESSENTIAL) << "countSmaller using " << TIMER_MSEC(constructMaximumBinaryTree) << " milliseconds";

    util::Log(logESSENTIAL) << "Running countSmaller tests:";
    TIMER_START(countSmaller);
    countSmaller_scaffold("[5]", "[0]");
    countSmaller_scaffold("[5,5,5,5]", "[0,0,0,0]");
    countSmaller_scaffold("[5,2,6,1]", "[2,1,1,0]");
    countSmaller_scaffold("[5,2,2,6,1]", "[3,1,1,1,0]");
    countSmaller_scaffold("[1,5,2,2,6,1]", "[0,3,1,1,1,0]");
    TIMER_STOP(countSmaller);
    util::Log(logESSENTIAL) << "countSmaller using " << TIMER_MSEC(countSmaller) << " milliseconds";

    util::Log(logESSENTIAL) << "Running sortArray tests:";
    TIMER_START(sortArray);
    sortArray_scaffold("[]", "[]");
    sortArray_scaffold("[5]", "[5]");
    sortArray_scaffold("[5,5,5,5]", "[5,5,5,5]");
    sortArray_scaffold("[5,2,6,1]", "[1,2,5,6]");
    sortArray_scaffold("[5,2,2,6,1]", "[1,2,2,5,6]");
    sortArray_scaffold("[1,2,3,4,5]", "[1,2,3,4,5]");
    TIMER_STOP(sortArray);
    util::Log(logESSENTIAL) << "sortArray using " << TIMER_MSEC(sortArray) << " milliseconds";
}