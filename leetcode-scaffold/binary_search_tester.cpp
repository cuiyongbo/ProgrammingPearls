#include "leetcode.h"

using namespace std;

static const int LOOP_COUNT = 100;

void binarySearchTester(int arraySize);
void lowerBoundSearchTester(int arraySize);
void upperBoundSearchTester(int arraySize);

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
        cout << "\tTestType=1 binary search test\n";
        cout << "\tTestType=2 lower bound search test\n";
        cout << "\tTestType=3 upper bound search test\n";
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
        else if(testType<0 || testType>3)
        {
            cout << "TestType must be choosen from 0,1,2,3\n";
            return 1;
        }
    }

    srand(time(NULL));

    if(testType == 1)
    {
        binarySearchTester(arraySize);
    }
    else if(testType == 2)
    {
        lowerBoundSearchTester(arraySize);
    }
    else if(testType == 3)
    {
        upperBoundSearchTester(arraySize);
    }
    else
    {
        binarySearchTester(arraySize);
        lowerBoundSearchTester(arraySize);
        upperBoundSearchTester(arraySize);
    }

    return 0;
}

void binarySearchTester(int arraySize)
{
    vector<int> input;

    auto binary_search_homebrew = [&] (int key)
    {
        bool found = false;
        int l = 0;
        int r = arraySize-1;
        while(l <= r)
        {
            int m = (r-l)/2 + l;
            if(input[m] == key)
            {
                found = true;
                break;
            }
            else if(input[m] < key)
            {
                l = m+1;
            }
            else
            {
                r = m-1;
            }
        }
        bool stdFound = binary_search(input.begin(), input.end(), key);
        if(found != stdFound)
        {
            cout << "binarySearchTester failed, array size: " << arraySize << endl;
        }
    };

    generateTestArray(input, arraySize, false);

    binary_search_homebrew(input[rand() % arraySize]);

    for (int i = 0; i < LOOP_COUNT; ++i)
    {
        binary_search_homebrew(rand());
    }

    generateTestArray(input, arraySize, true);
    binary_search_homebrew(input[0]);
    binary_search_homebrew(input[0]+1);
    binary_search_homebrew(input[0]-1);
}

void lowerBoundSearchTester(int arraySize)
{
    vector<int> input;
    auto lower_bound_homebrew = [&](int key) {
        int l = 0;
        int r = arraySize;
        while(l < r)
        {
            int m = (r-l)/2 + l;
            if(input[m] < key)
            {
                l = m+1;
            }
            else /*if(input[m] >= key)*/
            {
                r = m;
            }
        }

        const auto& it = lower_bound(input.begin(), input.end(), key);
        if(r != distance(input.begin(), it))
        {
            cout << "lowerBoundSearchTester failed, array size: " << arraySize << endl;
        }
    };

    generateTestArray(input, arraySize, false);

    lower_bound_homebrew(input[rand() % arraySize]);

    for(int i=0; i<LOOP_COUNT; ++i)
    {
        lower_bound_homebrew(rand());
    }

    generateTestArray(input, arraySize, true);
    lower_bound_homebrew(input[0]);
    lower_bound_homebrew(input[0] + 1);
    lower_bound_homebrew(input[0] - 1);
}

void upperBoundSearchTester(int arraySize)
{
    vector<int> input;
    auto upper_bound_homebrew = [&](int key) {
        int l = 0;
        int r = arraySize;
        while(l < r)
        {
            int m = (r-l)/2 + l;
            if(input[m] <= key)
            {
                l = m+1;
            }
            else /*if(input[m] > key)*/
            {
                r = m;
            }
        }

        const auto& it = upper_bound(input.begin(), input.end(), key);
        if(r != distance(input.begin(), it))
        {
            cout << "upperBoundSearchTester failed, array size: " << arraySize << endl;
        }
    };

    generateTestArray(input, arraySize, false);

    upper_bound_homebrew(input[rand() % arraySize]);

    for(int i=0; i<LOOP_COUNT; ++i)
    {
        upper_bound_homebrew(rand());
    }

    generateTestArray(input, arraySize, true);
    upper_bound_homebrew(input[0]);
    upper_bound_homebrew(input[0]+1);
    upper_bound_homebrew(input[0]-1);
}
