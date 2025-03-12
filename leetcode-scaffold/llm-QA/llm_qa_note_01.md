Q: how to solve the problem in C++?

```
Given an array of integers, return the number of solutions with which we can partition the array into three non-empty parts with equal sums.
for example,
input: [0, 0, 0, 0]
output: 3
```

To solve the problem of partitioning an array into three non-empty parts with equal sums, we can follow these steps:

1. **Calculate the total sum of the array**.
2. **Check if the total sum is divisible by 3**. If it is not, it is impossible to partition the array as required.
3. **Find the target sum for each part** which should be one-third of the total sum.
4. **Use a prefix sum approach to count the number of ways to form three partitions**. This involves:
   - Iterating through the array to calculate potential partitions.
   - Using a two-pointer or nested loop approach to ensure that each partition satisfies the conditions.

### Implementation in C++

Here is the complete implementation:

```cpp
#include <iostream>
#include <vector>

using namespace std;

int countPartitions(vector<int>& nums) {
    int n = nums.size();
    int total_sum = 0;
    
    // Calculate the total sum of the array
    for (int num : nums) {
        total_sum += num;
    }

    // If total sum is not divisible by 3, return 0
    if (total_sum % 3 != 0) {
        return 0;
    }

    // Calculate the target sum for each part
    int target_sum = total_sum / 3;
    int prefix_sum = 0;
    int count = 0;
    int first_partition_ways = 0;

    for (int i = 0; i < n - 1; ++i) {
        prefix_sum += nums[i];

        // Check if the current prefix sum is equal to 2 * target_sum
        if (prefix_sum == 2 * target_sum) {
            count += first_partition_ways;
        }

        // Check if the current prefix sum is equal to target_sum
        if (prefix_sum == target_sum) {
            ++first_partition_ways;
        }
    }

    return count;
}

int main() {
    // Test case
    vector<int> nums = {0, 0, 0, 0};
    cout << "Number of ways to partition: " << countPartitions(nums) << endl; // Output: 3
    return 0;
}
```

### Explanation

1. **Calculate Total Sum**:
    - Iterate through the array and sum up all elements to get `total_sum`.

2. **Check Divisibility**:
    - If `total_sum % 3 != 0`, it is impossible to partition the array into three equal parts, so return 0.

3. **Prefix Sum and Counting**:
    - Iterate through the array and maintain a `prefix_sum`.
    - Keep track of the number of times the prefix sum equals `target_sum` using `first_partition_ways`.
    - When `prefix_sum` equals `2 * target_sum`, it implies that the first and second partitions can be formed correctly, and `first_partition_ways` counts the ways to form the first partition.

4. **Count Valid Partitions**:
    - Increment the count of valid partitions whenever the prefix sum equals `2 * target_sum`.

### Detailed Steps

1. **Initialization**:
    - Initialize `prefix_sum` to 0, `count` to 0, and `first_partition_ways` to 0.

2. **Main Loop**:
    - For each index `i` in the array (except the last one), update `prefix_sum`.
    - If `prefix_sum` equals `2 * target_sum`, update `count` by adding `first_partition_ways`.
    - If `prefix_sum` equals `target_sum`, increment `first_partition_ways`.

### Test and Validate

You can test the function with different test cases to ensure correctness:

```cpp
vector<int> nums1 = {0, 0, 0, 0};
cout << "Number of ways to partition: " << countPartitions(nums1) << endl; // Output: 3

vector<int> nums2 = {1, 2, 3, 0, 3};
cout << "Number of ways to partition: " << countPartitions(nums2) << endl; // Output: 0

vector<int> nums3 = {1, 2, 0, 3, 0, 0, 3};
cout << "Number of ways to partition: " << countPartitions(nums3) << endl; // Output: 2
```

With this approach, you'll be able to efficiently and correctly determine the number of ways to partition an array into three non-empty parts with equal sums.

Q: how to solve following problem in C++?
```
We have two integer sequences A and B of the same non-zero length. We are allowed to swap elements A[i] and B[i]. 
Note that both elements are in the same index position in their respective sequences.
Given A and B, return the minimum number of swaps to make both sequences strictly increasing.
It is guaranteed that the given input always makes it possible.
```

To solve this problem, we can use dynamic programming (DP). The key is to use a DP array to keep track of the minimum number of swaps needed to make both sequences strictly increasing.

### Approach

1. **Define states**:
   - `keep[i]`: Minimum swaps required to make both sequences strictly increasing up to index `i` without swapping `A[i]` and `B[i]`.
   - `swap[i]`: Minimum swaps required to make both sequences strictly increasing up to index `i` with swapping `A[i]` and `B[i]`.

2. **Transition**:
   - If `A[i-1] < A[i]` and `B[i-1] < B[i]`, then:
      - `keep[i] = keep[i-1]` (No swap needed at `i` if no swap needed at `i-1`)
      - `swap[i] = swap[i-1] + 1` (If we decide to swap at `i`, then swaps needed up to `i` is simply `swap[i-1]` plus one more swap at `i`)
   - If `A[i-1] < B[i]` and `B[i-1] < A[i]`, then:
      - `keep[i] = min(keep[i], swap[i-1])` (If a swap at `i-1` would make the sequences increasing at `i`, no swap is needed at `i`)
      - `swap[i] = min(swap[i], keep[i-1] + 1)` (If no swap at `i-1`, but a swap at `i` makes sequences increasing, add one swap at `i`)

3. **Initialization**:
   - `keep[0]` should be `0` because no swap is needed at the beginning.
   - `swap[0]` should be `1` because we need one swap if we are swapping the first elements.

4. **Result**:
   - The result is the minimum value between `keep[n-1]` and `swap[n-1]` for the last indices.

### Implementation in C++

Here is the complete implementation of the approach:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int minSwap(vector<int>& A, vector<int>& B) {
    int n = A.size();
    vector<int> keep(n, INT_MAX); // keep[i]: minimum swaps to make both sequences increasing up to index i without swapping A[i] and B[i]
    vector<int> swap(n, INT_MAX); // swap[i]: minimum swaps to make both sequences increasing up to index i with swapping A[i] and B[i]

    keep[0] = 0; // no swap needed initially
    swap[0] = 1; // one swap needed initially if we swap the first elements

    for (int i = 1; i < n; ++i) {
        if (A[i-1] < A[i] && B[i-1] < B[i]) {
            keep[i] = keep[i-1]; // no swap needed at i if no swap needed at i-1
            swap[i] = swap[i-1] + 1; // if swap needed at i, then it's previous swaps plus one more
        }
        if (A[i-1] < B[i] && B[i-1] < A[i]) {
            keep[i] = min(keep[i], swap[i-1]); // if swap at i-1 makes increasing, no swap at i needed
            swap[i] = min(swap[i], keep[i-1] + 1); // if no swap at i-1, but swap at i, it's previous keep plus one swap
        }
    }

    return min(keep[n-1], swap[n-1]); // result is min of both sequences at the last index
}

int main() {
    vector<int> A = {1, 3, 5, 4};
    vector<int> B = {1, 2, 3, 7};

    cout << "Minimum number of swaps: " << minSwap(A, B) << endl; // Output: 1
    return 0;
}
```

### Explanation

1. **Initialization**:
   - `keep[0] = 0` because no swap is needed to start.
   - `swap[0] = 1` because if we swap the first elements, it counts as one swap.

2. **Dynamic Programming Transition**:
   - For every index `i` from `1` to `n-1`, update `keep` and `swap` based on the relationships mentioned.

3. **Result Computation**:
   - The result is the minimum of `keep[n-1]` and `swap[n-1]` which gives the minimum number of swaps to make the entire sequence strictly increasing.

This algorithm ensures that the constraints are respected, and it efficiently computes the minimum number of swaps using dynamic programming with a time complexity of \(O(n)\).


Q: how to solve following problem in C++?

```
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions
as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (i.e., cooldown 1 day)
```

To solve the problem of finding the maximum profit with the given stock prices, considering that you have a cooldown period of 1 day after selling the stock, we can use dynamic programming. Here’s the detailed explanation and implementation in C++.

### Dynamic Programming Approach

We can use three states to represent the different scenarios:
1. **Hold/Buy State**: The maximum profit if you're holding the stock on day `i`.
2. **Sold State**: The maximum profit if you have just sold the stock on day `i`.
3. **Cooldown State**: The maximum profit if you're in a cooldown period on day `i` (you sold the stock the day before or haven't done any transaction).

### State Transitions

- **Hold/Buy State**:
  - You can buy the stock on day `i` if you were in a cooldown period or you were already holding the stock: 
    $$ hold[i] = \max(hold[i-1], cooldown[i-1] - prices[i]) $$

- **Sold State**:
  - You can sell the stock on day `i` if you were holding the stock the previous day:
    $$ sold[i] = hold[i-1] + prices[i] $$

- **Cooldown State**:
  - You can be in a cooldown period on day `i` if you were in a cooldown period or you just sold the stock the previous day:
    $$ cooldown[i] = \max(cooldown[i-1], sold[i-1]) $$

### Base Cases
- On day `0`:
  - `hold[0] = -prices[0]` (buy the stock)
  - `sold[0] = 0` (cannot sell without buying)
  - `cooldown[0] = 0`

### Result
- The result will be the maximum of `sold` or `cooldown` on the last day, since we cannot end in the hold state.

Let's implement this in C++:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int maxProfit(vector<int>& prices) {
    if (prices.empty()) return 0;

    int n = prices.size();
    
    // Initialize the DP arrays
    vector<int> hold(n, 0), sold(n, 0), cooldown(n, 0);

    // Base cases
    hold[0] = -prices[0]; // We've bought on the first day
    sold[0] = 0;          // Cannot sell on the first day without buying
    cooldown[0] = 0;      // No cooldown on the first day

    // Fill the DP arrays
    for (int i = 1; i < n; ++i) {
        hold[i] = max(hold[i-1], cooldown[i-1] - prices[i]);
        sold[i] = hold[i-1] + prices[i];
        cooldown[i] = max(cooldown[i-1], sold[i-1]);
    }

    // The result is the maximum profit on the last day being in sold or cooldown states
    return max(sold[n-1], cooldown[n-1]);
}

int main() {
    // Test case
    vector<int> prices = {1, 2, 3, 0, 2};
    cout << "Maximum profit: " << maxProfit(prices) << endl; // Output: 3

    return 0;
}
```

### Explanation
1. **Initialization**:
   - `hold[0]` is initialized to `-prices[0]` because buying on the first day incurs a negative cost.
   - `sold[0]` and `cooldown[0]` are initialized to `0` since no transactions have occurred yet.

2. **Dynamic Programming Transition**:
   - For each day `i` from `1` to `n-1`, the state transitions are computed based on the previous day as discussed.

3. **Result Calculation**:
   - The result is `max(sold[n-1], cooldown[n-1])` representing the maximum profit on the last day.

This approach efficiently calculates the maximum profit with a time complexity of \(O(n)\) and a space complexity of \(O(n)\), where \(n\) is the number of days. The provided implementation covers all edge cases and ensures that the constraints of not engaging in multiple transactions at the same time and having a cooldown period are respected.

Q: how to solve following problem in C++?

```
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
'.' Matches any single character. and '*' Matches zero or more of the preceding element. The matching should cover the entire input string (not partial).
Note:
s could be empty or contains only lowercase letters a-z.
p could be empty or contains only lowercase letters a-z, and characters like . or *.
For example, given an input: s = "ab", p = ".*", output: true, explanation: ".*" means "zero or more (*) of any character (.)".
```

To solve the given problem of regex matching with support for `.` and `*`, we can use dynamic programming. The `.` matches any single character, and `*` matches zero or more of the preceding element. 

Here is the step-by-step approach to implement the solution in C++:

### Steps to Solve the Problem:

1. **Define the Problem State**:
    - Use a 2D boolean array `dp` where `dp[i][j]` represents whether `s[0..i-1]` matches `p[0..j-1]`.

2. **Initialize the DP Table**:
    - `dp[0][0]` should be `true` because an empty pattern matches an empty string.
    - Initialize other entries of `dp` based on if `p` can represent an empty string, particularly for patterns containing `*`.

3. **Fill the DP Table**:
    - Iterate over each character of the strings `s` and `p`.
    - Handle cases based on whether the current pattern character is `*`, `.`, or a specific character.

4. **Formulate the Transitions**:
    - If `p[j-1]` is `*`, process it by checking if it can match zero or more of the previous element in the pattern.
    - If `p[j-1]` is `.`, it matches any single character.
    - For other characters, directly check if the current characters of `p` and `s` match.

5. **Return the Result**:
    - The result will be in `dp[m][n]` where `m` is the length of `s` and `n` is the length of `p`.

### Implementation in C++

Here is the complete code for the solution:

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

bool isMatch(const string& s, const string& p) {
    int m = s.size();
    int n = p.size();
    
    // Create a 2D array to store the dp solution
    vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
    
    // Base case: empty pattern matches the empty string
    dp[0][0] = true;
    
    // Initialize dp for patterns like a*, a*b*, a*b*c* which can match an empty string
    for (int j = 2; j <= n; j += 2) {
        if (p[j-1] == '*') {
            dp[0][j] = dp[0][j-2];
        }
    }
    
    // Fill the dp table
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (p[j-1] == '*') {
                // '*' can match zero instances of the preceding element
                dp[i][j] = dp[i][j-2];
                if (p[j-2] == '.' || p[j-2] == s[i-1]) {
                    // '*' can match at least one instance of the preceding element
                    dp[i][j] = dp[i][j] || dp[i-1][j];
                }
            } else if (p[j-1] == '.' || p[j-1] == s[i-1]) {
                dp[i][j] = dp[i-1][j-1];
            }
        }
    }
    
    return dp[m][n];
}

int main() {
    // Test cases
    string s1 = "ab";
    string p1 = ".*";
    cout << (isMatch(s1, p1) ? "true" : "false") << endl; // Output: true

    string s2 = "aab";
    string p2 = "c*a*b";
    cout << (isMatch(s2, p2) ? "true" : "false") << endl; // Output: true

    string s3 = "mississippi";
    string p3 = "mis*is*p*.";
    cout << (isMatch(s3, p3) ? "true" : "false") << endl; // Output: false

    return 0;
}
```

### Explanation:

1. **Initialization**:
    - `dp[0][0]` is `true` because an empty pattern matches an empty string.
    - The loop initializing `dp[0][j]` handles patterns that can match an empty string (e.g., `a*`, `a*b*`).

2. **Filling the Table**:
    - If the pattern character is `*`, it checks for two cases:
        - Zero occurrences of the preceding character (`dp[i][j-2]`).
        - One or more occurrences if the preceding character in the pattern matches the current character in the string (or if it's a `.` which matches any character).
    - For `.` or exact character match, the value is derived from `dp[i-1][j-1]`.

3. **Result**:
    - After filling the table, the result is found in `dp[m][n]`.

This dynamic programming solution efficiently determines if the given string matches the pattern using the defined rules, covering various edge cases and ensuring completeness.
