#include <algorithm>
#include <string>
#include <iostream>
#include <cctype>
#include <vector>
 
int main()
{
    std::vector<int> vi;
    for(int i=0; i<20; i++)
        vi.push_back(i/4);
    std::cout << vi.size() << "/" << vi.capacity() << '\n';
    std::vector<int>::iterator iter2 = std::find(vi.begin(), vi.end(), 1);
    auto iter = std::remove(vi.begin(), iter2, 0);
    std::cout << std::distance(iter, iter2) << ',' << std::distance(vi.begin(), iter2) << '\n';
    std::cout << vi.size() << "/" << vi.capacity() << '\n';
    vi.erase(iter, iter2);
    std::cout << vi.size() << "/" << vi.capacity() << '\n';
}
