#include <thread>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
 
volatile int sink; // prevents optimization
 
void do_some_work(std::size_t size) {
    std::vector<int> v(size, 42);
    sink = std::accumulate(v.begin(), v.end(), 0); // ensures side effect
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}
 
int main() {
    std::cout << std::fixed << std::setprecision(9) << std::left;
    for (int64_t size=1; size < 1000000000; size *= 100) {
        const auto start = std::chrono::steady_clock::now();
        do_some_work(size);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff = end - start;
        std::cout << "diff = " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << "ms; size = " << size << '\n';
    }
}


/*
Q: how to convert `std::chrono::duration` into `std::chrono::second`?

In C++, you can convert `std::chrono::duration` to `std::chrono::seconds` by using the `std::chrono::duration_cast` function. Here's an example of how you can achieve this:

### C++ Example

To illustrate the conversion, consider an example where we convert various types of durations to seconds:

```cpp
#include <iostream>
#include <chrono>

int main() {
    // Define a duration of 1000 milliseconds
    std::chrono::milliseconds ms_duration(1000);

    // Convert milliseconds to seconds
    std::chrono::seconds sec_duration = std::chrono::duration_cast<std::chrono::seconds>(ms_duration);

    std::cout << "Milliseconds: " << ms_duration.count() << " ms" << std::endl;
    std::cout << "Seconds: " << sec_duration.count() << " s" << std::endl;

    // Define a duration of 2 minutes
    std::chrono::minutes min_duration(2);

    // Convert minutes to seconds
    sec_duration = std::chrono::duration_cast<std::chrono::seconds>(min_duration);

    std::cout << "Minutes: " << min_duration.count() << " min" << std::endl;
    std::cout << "Seconds: " << sec_duration.count() << " s" << std::endl;

    return 0;
}
```

### Explanation

1. **Include `<chrono>` Header**:
   - The header file `<chrono>` contains definitions for the duration and time point classes, as well as various utilities for dealing with time.

2. **Create a Duration Object**:
   - For example, `std::chrono::milliseconds ms_duration(1000);` creates a duration of 1000 milliseconds.

3. **Use `std::chrono::duration_cast` to Convert**:
   - To convert the duration to seconds, you use `std::chrono::duration_cast<std::chrono::seconds>(ms_duration);`.

4. **Print the Durations**:
   - Use `.count()` to get the numerical value of the duration.

This process applies to any duration type that `std::chrono` supports (e.g., nanoseconds, microseconds, milliseconds, minutes, hours, etc.). The `std::chrono::duration_cast` function allows you to convert between these different units in a consistent manner.

### Running Example Output

When you run the provided example code, you will see the following output:

```
Milliseconds: 1000 ms
Seconds: 1 s
Minutes: 2 min
Seconds: 120 s
```

This indicates that the duration was correctly converted from milliseconds and minutes to seconds, demonstrating the use of `std::chrono::duration_cast`.
*/