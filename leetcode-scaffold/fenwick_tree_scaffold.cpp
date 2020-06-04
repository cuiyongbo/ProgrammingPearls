#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>

using namespace std;

// https://iq.opengenus.org/fenwick-tree-binary-indexed-tree/

class FenwickTree
{
public:
    FenwickTree(const vector<int>& nums);

    // 1-indexed array
    void update(int i, int delta);
    int sumRange(int start, int end);

private:

    // get the least significant one
    int lsb(int x) { return x & (-x); } 

    // 1-indexed array
    int query(int x);

private:

    // m_ft[i] is responsible for elments in range [i-lsb(i)+1, i]
    vector<int> m_ft; 
};

FenwickTree::FenwickTree(const vector<int>& nums)
{
    int n = (int)nums.size();
    m_ft.assign(n+1, 0);
    for(int i=0; i<n; i++)
    {
        update(i+1, nums[i]);
    }
}

void FenwickTree::update(int pos, int delta)
{
    // Adding LSB(x) to x allows it to traverse 
    // its responsibility tree upwards.
    while(pos < (int)m_ft.size())
    {
        m_ft[pos] += delta;
        pos += lsb(pos);
    }
}

int FenwickTree::query(int x)
{
    // Subtracting LSB(x) from x gives the largest 
    // index that is not responsibility of x.
    
    int sum = 0;
    while(x > 0)
    {
        sum += m_ft[x];
        x -= lsb(x);
    }
    return sum;
}

int FenwickTree::sumRange(int start, int end)
{
    return query(end) - query(start-1);
}

int main()
{
    vector<int> nums(6);
    std::iota(nums.begin(), nums.end(), 1);
    std::cout << "nums: ";
    std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(cout, " "));
    std::cout << endl;
    
    FenwickTree ft(nums);
    cout << "sum[1,4]: " << ft.sumRange(1,4) << endl; 

    nums[3-1] += 7;
    std::cout << "nums: ";
    std::copy(nums.begin(), nums.end(), std::ostream_iterator<int>(cout, " "));
    std::cout << endl;

    ft.update(3, 7);
    cout << "sum[1,4]: " << ft.sumRange(1,4) << endl; 
}