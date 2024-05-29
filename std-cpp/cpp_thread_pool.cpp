#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>

// mimic `sync.WaitGroup` in golang
class DemoWaitGroup {
public:
    DemoWaitGroup() {
        m_count = 0;
    }

    void add() {
        inner_add(1);
    }

    void done() {
        inner_add(-1);
    }

    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond_var.wait(lock, [this](){ return m_count==0;});
    }

private:
    int inner_add(int n) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_count += n;
        if (m_count == 0) {
            m_cond_var.notify_all();
        }
        assert(m_count >= 0);
        return m_count;
    }

private:
    int m_count;
    std::mutex m_mutex;
    std::condition_variable m_cond_var;
};

using task_func_t =  std::function<void()>;
using task_func1_t = std::function<void(size_t i)>;
using task_func2_t = std::function<void(int tid, size_t i)>;

class ThreadPool {
public:
    ThreadPool(size_t thread_num) {
        m_thread_num = thread_num;
        m_stopping = false;
        start();
    }

    ~ThreadPool() {
        stop();
    }

    void enqueue(task_func_t&& task) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_task_queue.push(task);
        m_cond_var.notify_one();
    }

    template<class T>
    auto enqueue_future(T task) -> std::future<decltype(task())> {
        auto wrapper = std::make_shared<std::packaged_task<decltype(task())()>>(std::move(task));
        {
            std::unique_lock<std::mutex> lock{ m_mutex };
            m_task_queue.emplace([=] {
                (*wrapper)();
            });
        }
        m_cond_var.notify_one();
        return wrapper->get_future();
    }

private:
    void start() {
        for (int i=0; i<m_thread_num; i++) {
            m_workers.push_back(std::thread(&ThreadPool::worker_routine, this));
        }
    }

    void worker_routine() {
        while (true) {
            task_func_t one_task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cond_var.wait(lock, [this]() {return m_stopping || !m_task_queue.empty();});
                // 停止线程池前, 消耗完剩余任务
                if (m_stopping && m_task_queue.empty()) {
                    break;
                }
                one_task = std::move(m_task_queue.front());
                m_task_queue.pop();
            }
            one_task();
        }
    }

    void stop() {
        { // 通知 worker, 执行完池子里的任务后退出
            std::unique_lock<std::mutex> lock(m_mutex);
            m_stopping = true;
            m_cond_var.notify_all();
        }
        for (int i=0; i<m_workers.size(); i++) {
            m_workers[i].join();
        }
    }

private:
    bool m_stopping;
    int m_thread_num;
    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_cond_var;
    std::queue<task_func_t> m_task_queue;
};

void enqueue_future_test() {
    std::cout << "running " << __FUNCTION__ << std::endl;
    ThreadPool pool(4);
    auto f1 = pool.enqueue_future([] {
        std::cout << "Task 1 running\n";
        return 1;
    });
    auto f2 = pool.enqueue_future([] {
        std::cout << "Task 2 running\n";
        return 2;
    });
    auto f3 = pool.enqueue_future([] {
        std::cout << "Task 3 running\n";
        return "hello world";
    });
    std::cout << "Task 1 result: " << f1.get() << '\n';
    std::cout << "Task 2 result: " << f2.get() << '\n';
    std::cout << "Task 3 result: " << f3.get() << '\n';
}

void enqueue_test() {
    std::cout << "running " << __FUNCTION__ << std::endl;
    ThreadPool pool(4);
    pool.enqueue([] {
        std::cout << "Task 1 running\n";
    });
    pool.enqueue_future([] {
        std::cout << "Task 2 running\n";
    });
    pool.enqueue_future([] {
        std::cout << "Task 3 running\n";
    });
}

int main() {
    enqueue_future_test();
    enqueue_test();
    return 0;
}

/*
    how to compile a binary: g++ cpp_thread_pool.cpp -std=c++11
*/