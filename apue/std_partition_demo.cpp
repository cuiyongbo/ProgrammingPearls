#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <forward_list>
 
template <class ForwardIt>
void quicksort(ForwardIt first, ForwardIt last)
{
    if(first == last) return;
    auto pivot = *std::next(first, std::distance(first,last)/2);
    ForwardIt mid1 = std::partition(first, last, 
                         [pivot](const typename ForwardIt::value_type& em){ return em < pivot; });
    ForwardIt mid2 = std::partition(mid1, last, 
                         [pivot](const typename ForwardIt::value_type& em){ return !(pivot < em); });

    // [first, mid1) < pivot
    // [mid1, mid2) == pivot
    // [mid2, last) > pivot                  
    quicksort(first, mid1);
    quicksort(mid2, last);
}
 
int main()
{
    std::vector<int> v = {0,1,2,3,4,5,6,7,8,9};
    std::cout << "Original vector:\n    ";
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
 
    auto it = std::partition(v.begin(), v.end(), [](int i){return i % 2 == 0;});
    std::cout << "\nPartitioned vector:\n    ";
    std::copy(std::begin(v), it, std::ostream_iterator<int>(std::cout, " "));
    std::cout << " * ";
    std::copy(it, std::end(v), std::ostream_iterator<int>(std::cout, " "));
 
    std::vector<int> v1 = {0,1,2,3,4,5,6,7,8,9};
    auto it1 = std::stable_partition(v1.begin(), v1.end(), [](int i){return i % 2 == 0;});
    std::cout << "\nStable_partitioned vector:\n    ";
    std::copy(v1.begin(), it1, std::ostream_iterator<int>(std::cout, " "));
    std::cout << " * ";
    std::copy(it1, v1.end(), std::ostream_iterator<int>(std::cout, " "));
	
    std::forward_list<int> fl = {1, 30, -4, 3, 5, -4, 1, 6, -8, 2, -5, 64, 1, 92};
    std::cout << "\nUnsorted list:\n    ";
    std::copy(fl.begin(), fl.end(), std::ostream_iterator<int>(std::cout, " "));
 
    quicksort(std::begin(fl), std::end(fl));
    std::cout << "\nSorted using quicksort:\n    ";
    std::copy(fl.begin(), fl.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n'; 
}
