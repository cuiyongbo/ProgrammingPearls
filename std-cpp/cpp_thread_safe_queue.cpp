#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#include <queue>
#include <iterator>
#include <vector>
#include <map>
#include <unordered_map>

#include <iomanip>
#include <string>
#include <utility>
#include <type_traits>

#include <thread>
#include <chrono>

using namespace std;

class ThreadSafeQueue {
public:
    ThreadSafeQueue(int cap) {
        m_capacity = cap;
    }

    void push(int v) {
        std::unique_lock lock(m_mutex);
        m_cond_var.wait(lock, [this]() { return m_queue.size() < m_capacity; });
        m_queue.push(v);
        m_cond_var.notify_all();
    }

    int pop() {
        std::unique_lock lock(m_mutex);
        m_cond_var.wait(lock, [this]() { return !m_queue.empty(); });
        int val = m_queue.front();
        m_queue.pop();
        m_cond_var.notify_all();
        return val;
    }

    int capacity() const {
        return m_capacity;
    }

    int size() {
        std::unique_lock lock(m_mutex);
        return m_queue.size();
    }

    bool empty() {
        std::unique_lock lock(m_mutex);
        return m_queue.empty();
    }

    bool full() {
        std::unique_lock lock(m_mutex);
        return m_queue.size() == m_capacity;
    }

private:
    int m_capacity;
    std::mutex m_mutex;
    std::condition_variable m_cond_var;
    std::queue<int> m_queue;
};

ThreadSafeQueue global_plates(5);
std::mutex global_cout_mutex;

void plate_fn(bool op_add) {
    for (int i=0; i<10; i++) {
        if (op_add) {
            global_plates.push(i);
        } else {
            global_plates.pop();
        }
        {
            std::scoped_lock guard(global_cout_mutex);
            cout << std::this_thread::get_id() << ": " << global_plates.size() << endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(random()%100));
    }
}


int main(int argc, char* argv[]) {
    srandom(0);

    int thread_count = 10;
    vector<std::thread> thread_list;
    for (int i=0; i<thread_count; i++) {
        thread_list.push_back(std::thread(plate_fn, i%2==0));
    }
    for (int i=0; i<thread_count; i++) {
        thread_list[i].join();
    }
}


/*
Q: how to implement a thread-safe queue with limited capacity in c++?

To implement a thread-safe queue with limited capacity in C++, you can use the standard library components like `std::queue` for the queue data structure, along with synchronization primitives like `std::mutex` and `std::condition_variable` to ensure thread safety and manage concurrent access.

Here's a simple example of how you might implement such a queue:

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    size_t capacity;

public:
    explicit ThreadSafeQueue(size_t capacity_) : capacity(capacity_) {}

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        not_full.wait(lock, [this]() { return queue.size() < capacity; });
        queue.push(item);
        not_empty.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        not_empty.wait(lock, [this]() { return !queue.empty(); });
        T item = queue.front();
        queue.pop();
        not_full.notify_one();
        return item;
    }

    bool try_push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.size() < capacity) {
            queue.push(item);
            not_empty.notify_one();
            return true;
        }
        return false;
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {
            return false;
        }
        item = queue.front();
        queue.pop();
        not_full.notify_one();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    bool full() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size() >= capacity;
    }
};
```

This `ThreadSafeQueue` class template provides the following features:

- `push`: Adds an item to the queue. If the queue is full, it waits until there is space.
- `pop`: Removes and returns an item from the queue. If the queue is empty, it waits until there is an item to remove.
- `try_push`: Tries to add an item to the queue without waiting. Returns `true` if successful, `false` if the queue is full.
- `try_pop`: Tries to remove an item from the queue without waiting. Returns `true` if successful, `false` if the queue is empty.
- `size`: Returns the number of items in the queue.
- `empty`: Checks if the queue is empty.
- `full`: Checks if the queue is full.

The `std::condition_variable` is used to block threads when the queue is full (for `push`) or empty (for `pop`) and to wake up waiting threads when the queue state changes.

Please note that this is a simple implementation and may need to be adapted to fit the specific requirements of your application, such as handling exceptions or providing different synchronization mechanisms.

Q: can we use only one `condition_variable` in `ThreadSafeQueue`?

Yes, you can use a single `std::condition_variable` in a `ThreadSafeQueue` implementation. However, you need to be careful with the logic that handles the waiting and notification of threads, as you'll be using the same condition variable for both full and empty conditions.

Here's an example of how you might modify the `ThreadSafeQueue` class to use only one `std::condition_variable`:

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable condition;
    size_t capacity;

public:
    explicit ThreadSafeQueue(size_t capacity_) : capacity(capacity_) {}

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this]() { return queue.size() < capacity; });
        queue.push(item);
        condition.notify_all(); // Notify all waiting threads
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this]() { return !queue.empty(); });
        T item = queue.front();
        queue.pop();
        condition.notify_all(); // Notify all waiting threads
        return item;
    }

    bool try_push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.size() < capacity) {
            queue.push(item);
            condition.notify_one(); // Notify one waiting thread
            return true;
        }
        return false;
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {
            return false;
        }
        item = queue.front();
        queue.pop();
        condition.notify_one(); // Notify one waiting thread
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    bool full() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size() >= capacity;
    }
};
```

In this modified version, the `condition` variable is used for both waiting for space to push and waiting for items to pop. When a thread pushes an item or pops an item from the queue, it calls `condition.notify_all()` to wake up all waiting threads. Those threads will then re-evaluate their respective conditions (`queue.size() < capacity` for push and `!queue.empty()` for pop) to determine if they can proceed or need to wait again.

Using `notify_all()` instead of `notify_one()` ensures that both producers and consumers are notified when there is a change in the queue's state. However, this can lead to some inefficiency because it wakes up all waiting threads, even if only one can proceed. This is known as the "thundering herd problem." Depending on the specific use case and performance requirements, you may want to consider more sophisticated mechanisms to minimize unnecessary wake-ups.


Q: but why we call `condition .notify_one` in `try_push` and `try_pop` then?

You're correct that in the context of a single `std::condition_variable` being used for both conditions, it would be more consistent to use `condition.notify_all()` in both `try_push` and `try_pop` methods as well. This ensures that all waiting threads are notified of the change in the queue's state, allowing them to re-evaluate their wait conditions.

The reason for using `condition.notify_one()` in the `try_push` and `try_pop` methods in the original example was based on the assumption that there were separate condition variables for "not full" and "not empty" conditions. In that scenario, `notify_one()` would be sufficient because you would only need to wake up one thread that could potentially make progress (either a producer waiting to push when the queue is not full, or a consumer waiting to pop when the queue is not empty).

However, with a single `std::condition_variable`, you would typically use `notify_all()` to wake up all threads, as any change in the queue could potentially affect both producers and consumers. Here's the corrected version of the `try_push` and `try_pop` methods using `notify_all()`:

```cpp
bool try_push(const T& item) {
    std::lock_guard<std::mutex> lock(mutex);
    if (queue.size() < capacity) {
        queue.push(item);
        condition.notify_all(); // Notify all waiting threads
        return true;
    }
    return false;
}

bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!queue.empty()) {
        item = queue.front();
        queue.pop();
        condition.notify_all(); // Notify all waiting threads
        return true;
    }
    return false;
}
```

Using `notify_all()` in these methods ensures that all threads that are potentially waiting on the condition variable are notified and have a chance to proceed if the queue state allows it. This change makes the behavior consistent across all methods that modify the queue's state.


Q: in above example we call `wait` and `notify_all` in pair, is it necessary?

The calls to `wait` and `notify_all` (or `notify_one`) serve different purposes in a thread-safe queue, and they are not necessarily always used in pairs. Here's what each is used for:

- `wait`: This is used to block the current thread until a certain condition is met. In the context of a thread-safe queue, `wait` is used to block a thread trying to `pop` from an empty queue until there is an item available, or to block a thread trying to `push` to a full queue until there is space available. The thread will remain blocked until another thread modifies the queue and calls `notify_one` or `notify_all`.

- `notify_one` or `notify_all`: These are used to wake up one or more threads that are waiting on the condition variable. In the context of a thread-safe queue, you call `notify_one` or `notify_all` after pushing an item to the queue (to wake up threads waiting to `pop`) or after popping an item from the queue (to wake up threads waiting to `push`).

It is not strictly necessary to call `notify_all` after every `wait`. The key is to ensure that `notify_one` or `notify_all` is called whenever there is a change in the queue that could potentially allow a waiting thread to proceed. For example:

- When an item is added to the queue, you should notify waiting threads because this could allow a thread that is waiting to `pop` to proceed.
- When an item is removed from the queue, you should notify waiting threads because this could allow a thread that is waiting to `push` (due to the queue being full) to proceed.

The choice between `notify_one` and `notify_all` depends on the specific use case:

- `notify_one` is typically used when you know that only one waiting thread can make progress (e.g., when you have separate condition variables for "not full" and "not empty" conditions).
- `notify_all` is used when multiple waiting threads might be able to make progress, or when you have a single condition variable for both "not full" and "not empty" conditions, as in the modified example.

In summary, `wait` is called when a thread needs to wait for a condition to be met, and `notify_one`/`notify_all` is called when there is a change in the state that might allow waiting threads to proceed. They work together to synchronize access to the queue and to coordinate between producer and consumer threads.

*/