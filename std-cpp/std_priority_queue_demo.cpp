#include <iostream>
#include <functional>
#include <vector>
#include <queue>

using namespace std;

template<typename T> 
void printQueue(T& q) {
	while(!q.empty()) {
		cout << q.top() << ' ';
		q.pop();
	}
	cout << '\n';
}

int main() {
	// max-heap, std::less<int> means subnode < root
	priority_queue<int> q;
	for(int n: {1,8,5,6,3,4,0,9,7,2}) {
		q.push(n);
	}
	
	printQueue(q);
	
	// min-heap
	priority_queue<int, vector<int>, greater<int> > q2;
	for(int n: {1,8,5,6,3,4,0,9,7,2}) {
		q2.push(n);
	}

	printQueue(q2);

	auto cmp = [](int left, int right) {return (left^1) < (right^1);};
	priority_queue<int, vector<int>, decltype(cmp) > q3(cmp);
	for(int n: {1,8,5,6,3,4,0,9,7,2}) {
		q3.push(n);
	}

	printQueue(q3);
}


