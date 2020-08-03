#include <iostream>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>

using std::cout;
using std::string;

std::map<std::string, std::string> g_pages;
std::mutex g_pages_mutex;
 
void save_page(const string& url) {
    // simulate a long page fetch
    std::this_thread::sleep_for(std::chrono::seconds(2));
    string result = "xxxxooooo";
    std::lock_guard<std::mutex> guard(g_pages_mutex);
    g_pages[url] = result;
}

void page_fetcher() {
    std::thread t1(save_page, "http://foo");
    std::thread t2(save_page, "http://bar");
    t1.join();
    t2.join();

    for (const auto& p: g_pages) {
        cout << p.first << ": " << p.second << '\n';
    }
}

int g_i = 0;
std::mutex g_i_mutex;

void safe_increment() {
    std::lock_guard<std::mutex> guard(g_i_mutex);
    g_i++;
    cout << std::this_thread::get_id() << ": " << g_i << '\n';
}

void random_increment() {
    std::thread t1(safe_increment);
    std::thread t2(safe_increment);
    t1.join(); t2.join();
    cout << "random_increment: " << g_i << '\n';
}


int main(int argc, char* argv[]) {
    page_fetcher();
    random_increment();
}
