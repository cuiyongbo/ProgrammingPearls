#include "leetcode.h"

template<class UnaryPredicate>
int partition_private(std::vector<int>& vi, UnaryPredicate p)
{
    int i = -1;
    int size = (int)vi.size();
    for(int j=0; j<size; j++)
    {
        if(p(vi[j]))
        {
            ++i;
            std::swap(vi[i], vi[j]);
        }
    }
    return i+1; 
}

template< class BidirIt, class UnaryPredicate >
BidirIt partition_std( BidirIt first, BidirIt last, UnaryPredicate p)
{
    first = std::find_if_not(first, last, p);
    if(first == last) return first;
    for(auto it = std::next(first); it != last; ++it)
    {
        if(p(*it))
        {
            std::iter_swap(it, first);
            ++first;
        }
    }
    return first;
}

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cout << "prog arraySize\n";
        return 1;
    }
    
    srand(time(NULL));

    std::vector<int> input;
    int arraySize = std::atoi(argv[1]);
    generateTestArray(input, arraySize, false, false);

    int k = input[rand()%arraySize];

    auto it = partition_std(input.begin(), input.end(), std::bind2nd(std::less<int>(), k));
    int offset1 = (int)std::distance(input.begin(), it);

    std::random_shuffle(input.begin(), input.end());
    int offset2 = partition_private(input, std::bind2nd(std::less<int>(), k));

    assert(offset1 == offset2 && "partition test failed");

    return 0;
}


