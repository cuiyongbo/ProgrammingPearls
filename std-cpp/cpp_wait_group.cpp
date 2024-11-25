#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <future>
#include <iostream>


// mimic `sync.WaitGroup` in golang
// [waitgroup demo](https://gobyexample.com/waitgroups)
class DemoWaitGroup {
public:
    DemoWaitGroup() {
        m_count = 0;
    }

    ~DemoWaitGroup() {
        wait();
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

    // disable copy and assign operation
    DemoWaitGroup(const DemoWaitGroup&) = delete;
    DemoWaitGroup& operator=(const DemoWaitGroup&) = delete;

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


void worker_fn(int id, DemoWaitGroup* group) {
    printf("Worker %d starting\n", id);
    // sleep
    std::this_thread::sleep_for(std::chrono::seconds(5));
    printf("Worker %d done\n", id);
    group->done();
}

int main() {
    DemoWaitGroup group;
    int thread_count = 5;
    std::vector<std::thread> workers;
    for (int i=0; i<thread_count; i++) {
        group.add();
        std::thread t(worker_fn, i, &group);
        workers.push_back(std::move(t));
    }
    group.wait();
    /* 
    without calling `thread::join`, the program would abort:
        libc++abi: terminating
        Abort trap: 6
    */
    for (int i=0; i<thread_count; i++) {
        workers[i].join();
    }
    return 0;
}

/*
    how to compile a binary: g++ cpp_thread_pool.cpp -std=c++11
*/

/*
// To wait for multiple goroutines to finish, we can
// use a *wait group*.

package main

import (
	"fmt"
	"sync"
	"time"
)

// This is the function we'll run in every goroutine.
func worker(id int) {
	fmt.Printf("Worker %d starting\n", id)

	// Sleep to simulate an expensive task.
	time.Sleep(time.Second)
	fmt.Printf("Worker %d done\n", id)
}

func main() {

	// This WaitGroup is used to wait for all the
	// goroutines launched here to finish. Note: if a WaitGroup is
	// explicitly passed into functions, it should be done *by pointer*.
	var wg sync.WaitGroup

	// Launch several goroutines and increment the WaitGroup
	// counter for each.
	for i := 1; i <= 5; i++ {
		wg.Add(1)

		// Wrap the worker call in a closure that makes sure to tell
		// the WaitGroup that this worker is done. This way the worker
		// itself does not have to be aware of the concurrency primitives
		// involved in its execution.
		go func() {
			defer wg.Done()
			worker(i)
		}()
	}

	// Block until the WaitGroup counter goes back to 0;
	// all the workers notified they're done.
	wg.Wait()

	// Note that this approach has no straightforward way
	// to propagate errors from workers. For more
	// advanced use cases, consider using the
	// [errgroup package](https://pkg.go.dev/golang.org/x/sync/errgroup).
}
*/