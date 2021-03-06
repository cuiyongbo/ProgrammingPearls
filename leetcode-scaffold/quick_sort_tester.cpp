#include "leetcode.h"

using namespace std;

typedef int (*partition_func_t)(vector<int>&, int, int);
int partitioner(vector<int>&, int, int);
int randomized_naive_partitioner(vector<int>&, int, int);
int hoare_partitioner(vector<int>&, int, int);

void quickSort(vector<int>& input, partition_func_t partition);
void quickSort(vector<int>& input, int start, int end, partition_func_t partition); // input[start, end)

void quickSort_hoare(vector<int>& input);

template<class ForwardIter>
void quickSort_std(ForwardIter first, ForwardIter last);

// Hint: stack the ranges to be sort
void quickSort_iterative(vector<int>& input)
{
    stack<pair<int, int>> st;
    st.emplace(0, input.size());
    while(!st.empty())
    {
        auto t = st.top(); st.pop();
        int m = partitioner(input, t.first, t.second);
        if(t.first < m) st.emplace(t.first, m);
        if(m+1 < t.second) st.emplace(m+1, t.second);
    }
}

int main(int argc, char* argv[])
{
    int arraySize = 0;
    int testType = 0;
    string path(argv[0]);
    string programName = path.substr(path.find_last_of('/')+1);
    if(argc != 3)
    {
        cout << "Usage: " << programName << " ArraySize" << " TestType\n" ;
        cout << "\tArraySize must be positive\n";
        cout << "\tTestType=0 test all\n";
        cout << "\tTestType=1 quickSort with naive partitioner\n";
        cout << "\tTestType=2 quickSort with randomized naive partitioner\n";
        cout << "\tTestType=3 quickSort with hoare partitioner\n";
        cout << "\tTestType=4 quickSort with std partitioner\n";
        return 1;
    }
    else
    {
        arraySize = atoi(argv[1]);
        testType = atoi(argv[2]);
        if(arraySize <= 0)
        {
            cout << "ArraySize must be positive!\n";
            return 1;
        }
        else if(testType<0 || testType>4)
        {
            cout << "TestType must be choosen from 0,1,2,3,4\n";
            return 1;
        }
    }

    srand(time(NULL));

    vector<int> input;
    generateTestArray(input, arraySize, false, false);

    if(testType == 0)
    {
        quickSort(input, partitioner);
        quickSort(input, randomized_naive_partitioner);
        quickSort_hoare(input);
        quickSort_std(input.begin(), input.end());
    }
    else if(testType == 1)
    {
        quickSort(input, partitioner);
    }
    else if(testType == 2)
    {
        quickSort(input, randomized_naive_partitioner);
    }
    else if(testType == 3)
    {
        quickSort_hoare(input);
    }
    else if(testType == 4)
    {
        quickSort_std(input.begin(), input.end());
    }
    
    assert(is_sorted(input.begin(), input.end()) && "quickSort failed");
    
    return 0;
}

void quickSort(vector<int>& input, partition_func_t partition)
{
    return quickSort(input, 0, input.size(), partition);
}

void quickSort(vector<int>& input, int start, int end, partition_func_t partition)
{
    if(start + 1 >= end)
        return;

    int m = partition(input, start, end);
    quickSort(input, start, m, partition);
    quickSort(input, m+1, end, partition);
}

int partitioner(vector<int>& input, int start, int end)
{
    int i = start - 1;
    int k = input[end-1];
    for(int j=i+1; j<end; ++j)
    {
        if(input[j] < k)
        {
            swap(input[j], input[++i]);
        }
    }
    swap(input[i+1], input[end-1]);
    return i+1;
}

int randomized_naive_partitioner(vector<int>& input, int start, int end)
{
    int p = start + rand() % (end - start);
    swap(input[p], input[end-1]);
    return partitioner(input, start, end);
}

int hoare_partitioner(vector<int>& input, int start, int end)
{
    int p = rand() % (end - start) + start;
    swap(input[p], input[start]);
    int k = input[start];
    int i = start - 1;
    int j = end;

    // invariant: input[start, i] <= k and input[j, end) >= k
    while(i < j)
    {
        while(true)
        {
            if(input[--j] <= k) break;
        }

        while(true)
        {
            if(input[++i] >= k) break;
        }

        // input[j] <= k and input[i] >= k
        if(i < j)
        {
            swap(input[i], input[j]);
        }
    }
    return j;
}

void quickSort_hoare(vector<int>& input)
{
    function<void(int, int)> workhorse = [&](int start, int end)
    {
        if(start + 1 >= end)
            return;

        int m = hoare_partitioner(input, start, end);
        workhorse(start, m+1);
        workhorse(m+1, end);
    };

    return workhorse(0, input.size());
}

template<class ForwardIter>
void quickSort_std(ForwardIter first, ForwardIter last)
{
    size_t count = std::distance(first, last);
    if(count < 2)
        return;

    auto p = std::next(first, count/2);
    auto mid1 = partition(first, last, std::bind2nd(less<typename ForwardIter::value_type>(), *p));
    auto mid2 = partition(mid1, last, std::bind2nd(equal_to<typename ForwardIter::value_type>(), *p));

    quickSort_std(first, mid1);
    quickSort_std(mid2, last);
}
