#include "leetcode.h"

using namespace std;

static const int LOOP_COUNT = 100;

void binarySearchTester(int arraySize);
void lowerBoundSearchTester(int arraySize);
void upperBoundSearchTester(int arraySize);

int main(int argc, char* argv[]) {
    int arraySize = 0;
    int testType = 0;
    string path(argv[0]);
    string programName = path.substr(path.find_last_of('/')+1);
    if (argc != 3) {
        cout << "Usage: " << programName << " ArraySize" << " TestType\n" ;
        cout << "\tArraySize must be a positive integer\n";
        cout << "\tTestType=0 test all\n";
        cout << "\tTestType=1 binary search test\n";
        cout << "\tTestType=2 lower bound search test\n";
        cout << "\tTestType=3 upper bound search test\n";
        return 1;
    } else {
        arraySize = atoi(argv[1]);
        testType = atoi(argv[2]);
        if (arraySize <= 0) {
            cout << "ArraySize must be a positive integer!\n";
            return 1;
        } else if (testType<0 || testType>3) {
            cout << "TestType must be choosen from 0,1,2,3\n";
            return 1;
        }
    }

    srand(1234);

    if (testType == 1) {
        binarySearchTester(arraySize);
    } else if (testType == 2) {
        lowerBoundSearchTester(arraySize);
    } else if (testType == 3) {
        upperBoundSearchTester(arraySize);
    } else {
        binarySearchTester(arraySize);
        lowerBoundSearchTester(arraySize);
        upperBoundSearchTester(arraySize);
    }

    return 0;
}

void binarySearchTester(int arraySize) {
    auto worker = [&] (const vector<int>& input, int key) {
        bool found = false;
        int l = 0;
        int r = input.size() - 1;
        while (l <= r) {
            int m = (l+r)/2;
            if (input[m] == key) {
                found = true;
                break;
            } else if (input[m] < key) {
                l = m+1;
            } else {
                r = m-1;
            }
        }
        bool expectedResult = std::binary_search(input.begin(), input.end(), key);
        if (found != expectedResult) {
            printf("binarySearchTester failed, arraySize: %d, expected result: %d, acutal result: %d\n", arraySize, expectedResult, found);
            abort();
        }
    };

    vector<int> input;
    generateTestArray(input, arraySize, false, true);
    for (int i = 0; i < LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    generateTestArray(input, arraySize, true, true);
    for (int i = 0; i < LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
}

void lowerBoundSearchTester(int arraySize) {
    auto worker = [&](const vector<int>& input, int key) {
        int l=0;
        int r = input.size();
        while (l < r) {
            int m = (l+r)/2;
            if (input[m] < key) {
                l = m+1;
            } else {
                r = m;
            }
        }
        auto it = std::lower_bound(input.begin(), input.end(), key);
        int expectedResult = std::distance(input.begin(), it);
        if (l != expectedResult) {
            printf("lowerBoundSearchTester failed, arraySize: %d, key: %d, expected result: %d, acutal result: %d, %d\n", arraySize, key, expectedResult, l, r);
            abort();
        }
    };

    vector<int> input;
    generateTestArray(input, arraySize, false, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    generateTestArray(input, arraySize, true, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
}

void upperBoundSearchTester(int arraySize) {
    auto worker = [&](const vector<int>& input, int key) {
        int l=0;
        int r = input.size();
        while (l < r) {
            int m = (l+r)/2;
            if (input[m] <= key) {
                l = m+1;
            } else {
                r = m;
            }
        }
        auto it = std::upper_bound(input.begin(), input.end(), key);
        int expectedResult = std::distance(input.begin(), it);
        if (l != expectedResult) {
            printf("upperBoundSearchTester failed, arraySize: %d, key: %d, expected result: %d, acutal result: %d, %d\n", arraySize, key, expectedResult, l, r);
            abort();
        }
    };
    vector<int> input;
    generateTestArray(input, arraySize, false, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
    generateTestArray(input, arraySize, true, true);
    for(int i=0; i<LOOP_COUNT; ++i) {
        int ri = rand()%arraySize;
        worker(input, input[ri]);
        worker(input, rand());
    }
}