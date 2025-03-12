#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>

/*
You just need to do `std::hash<std::thread::id>{}(std::this_thread::get_id())` to get a size_t.

From cppreference:

> The template specialization of std::hash for the std::thread::id class allows users to obtain hashes of the identifiers of threads.
*/

size_t get_thread_id() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

/*
    how to compile a binary: g++ cpp_thread_pool.cpp -std=c++11
*/

std::mutex cout_mutex;
using task_func_t = std::function<void()>;
// typedef void (*task_func_t)(void);


class ThreadPool {
private:
    int m_worker_num;
    std::vector<std::thread> m_workers;
    std::queue<task_func_t> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_is_running;

public:
    ThreadPool(int worker_num) {
        m_worker_num = worker_num;
        m_is_running = true;
        start();
    }

    ~ThreadPool() {
        for (int i=0; i<m_worker_num; i++) {
            m_workers[i].join();
        }
    }

    int task_count() {
        // you must query the size of queue with lock obtained
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_tasks.size();
    }

    template<class F, class... Args>
    int enqueue(F&& f, Args&&... args) {
        // push tasks into the queue
        return enqueue_impl(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }

    int enqueue_impl(task_func_t&& t) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_is_running) {
            std::cout << "thread pool is going to close, no more tasks" << std::endl;
            return 1;
        }
        m_tasks.push(t);
        m_cv.notify_one();
        return 0;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_is_running = false;
        m_cv.notify_all();
    }

private:
    void woker_fn() {
        while (true) {
            task_func_t task;
            {
                // wait with lock obtained
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this](){
                    return !m_tasks.empty() || !m_is_running;
                });
                if (m_tasks.empty() && !m_is_running) {
                    break;
                }
                auto t = m_tasks.front(); m_tasks.pop();
                task = std::move(t); // task must be moved
            }
            task();
        }
    }

    void start() {
        for (int i=0; i<m_worker_num; i++) {
            // pass class member function as thread worker
            std::thread t = std::thread(&ThreadPool::woker_fn, this);
            m_workers.push_back(std::move(t)); // threads must be moved
        }
    }
};


int main(void) {
    ThreadPool pool(5);
    for (int i=0; i<20; i++) {
        pool.enqueue([](){
            std::unique_lock<std::mutex> lock(cout_mutex);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            printf("thread_id: %#zx with func()\n", get_thread_id());
        });
        pool.enqueue([](int id){
            std::unique_lock<std::mutex> lock(cout_mutex);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            printf("thread_id: %#zx with func(%d)\n", get_thread_id(), id);
        }, i);
        pool.enqueue(
            [](int x, int y) {
                std::unique_lock<std::mutex> lock(cout_mutex);
                printf("thread_id: %#zx with func(%d, %d)\n", get_thread_id(), x, y);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            },
            i, 10*i
        );
    }
    pool.stop();
    std::cout << "task_count: " << pool.task_count() << std::endl;
    pool.enqueue(
        [](int x, int y) {
            std::unique_lock<std::mutex> lock(cout_mutex);
            printf("thread_id: %#zx with func(%d, %d)\n", get_thread_id(), x, y);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        },
        1, 2
    );
}

/*
Implementing a thread pool in C++ involves creating a certain number of threads that are kept alive to perform various tasks as they're provided. This is a common way to manage resources and improve performance for concurrent tasks.

Here is a high-level overview of how to implement a basic thread pool:

1. **Create a work queue**: This queue holds all the tasks that need to be executed. Tasks can be represented as `std::function<void()>` objects or any callable type.

2. **Create worker threads**: Create a specific number of threads that will process tasks from the work queue.

3. **Synchronize access to the work queue**: Since multiple threads access the queue, you need synchronization mechanisms like mutexes and condition variables to ensure thread-safe operations.

4. **Implement a thread loop**: Each thread runs a loop that retrieves tasks from the work queue and executes them.

5. **Manage thread lifecycle**: Implement thread joining or detachment, and make sure that no threads are left hanging on program exit.

6. **Handle shutdown**: Provide a method for shutting down the thread pool gracefully, ensuring all tasks are completed before closing.

Below is a simple example of a thread pool implementation in C++ using the C++ Standard Library:

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
public:
    ThreadPool(size_t threadCount) : done(false) {
        for (size_t i = 0; i < threadCount; ++i) {
            workers.emplace_back([this]{
                while (!done) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this]{ return done || !taskQueue.empty(); });
                        if (done && taskQueue.empty())
                            return;
                        task = std::move(taskQueue.front());
                        taskQueue.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            done = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
      
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            if(done)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            taskQueue.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> taskQueue;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool done;
};

// Usage example
void exampleTask() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Task is executed." << std::endl;
}

int main() {
    ThreadPool pool(4); // Create a thread pool with 4 threads

    // Enqueue a couple of tasks
    for (int i = 0; i < 8; ++i) {
        pool.enqueue(exampleTask);
    }

    // Destructor of ThreadPool will be called here
    // All tasks should be completed before main() exits and pool is destroyed.
    return 0;
}
```

This is a very simple implementation and does not cover all edge cases and possible optimizations. A real-world thread pool might have additional features, such as task prioritization, better exception handling, dynamic thread creation, and more sophisticated shutdown procedures.

Always remember, in production code, you would also want to make sure that you properly handle tasks that throw exceptions, to prevent threads from exiting prematurely.

The use of `condition.notify_one()` in `enqueue()` and `condition.notify_all()` in `~ThreadPool()` has to do with the intended behavior in those two different scenarios within the thread pool.

1. **In `enqueue()`:**
   When we add a new task to the queue, we only need to wake up one waiting thread to handle the new task. This is more efficient than waking up all threads because only one thread is required to process one task at a time. If multiple threads were woke up, they would all try to lock the mutex and access the queue, but only one could proceed, leading to unnecessary contention. Using `condition.notify_one()` is an optimization to ensure that only one thread is awoken to take the new task, reducing contention and context switching overhead.

2. **In `~ThreadPool()`:**
   When destroying the pool (`~ThreadPool()`), we set the `done` flag to `true`, which signals to all worker threads that they should not wait for new tasks anymore and should exit their run loop so that they can be joined in the destructor. In this situation, we use `condition.notify_all()` because we want to wake up all the waiting threads. If we only woke up one thread, only that single thread would see `done == true` and exit, while the others would remain blocked on `condition.wait()`. `condition.notify_all()` ensures all threads wake up and thus can exit the waiting state to check the `done` flag, allowing the destructor to join them properly.

In summary, `condition.notify_one()` is an efficiency measure to wake only one thread for a new task during normal pool operation, and `condition.notify_all()` is used at destruction time to wake all threads to ensure they all exit and can be joined back into the main thread.

*/