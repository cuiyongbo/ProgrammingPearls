#include <thread>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <ctime>
 
// https://stackoverflow.com/questions/52421819/does-steady-clocknow-return-seconds

volatile int sink; // prevents optimization
 
void do_some_work(std::size_t size) {
    std::vector<int> v(size, 42);
    sink = std::accumulate(v.begin(), v.end(), 0); // ensures side effect
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void test_timestamp() {
    std::cout << "running " << __FUNCTION__ << std::endl;
    const auto p0 = std::chrono::time_point<std::chrono::system_clock>{};
    const auto p1 = std::chrono::system_clock::now();
    const auto p2 = p1 - std::chrono::hours(24);
 
    std::time_t epoch_time = std::chrono::system_clock::to_time_t(p0);
    std::cout << "epoch: " << std::ctime(&epoch_time);
    std::time_t today_time = std::chrono::system_clock::to_time_t(p1);
    std::cout << "today: " << std::ctime(&today_time);
 
    std::cout << "hours since epoch: "
              << std::chrono::duration_cast<std::chrono::hours>(
                     p1.time_since_epoch()).count() 
              << '\n';
    std::cout << "yesterday, hours since epoch: "
              << std::chrono::duration_cast<std::chrono::hours>(
                     p2.time_since_epoch()).count() 
              << '\n';
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

    test_timestamp();
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

This indicates that the duration was correctly converted from milliseconds and minutes to seconds, demonstrating the use of `std::chrono::duration_cast`.\

Q: what is the difference between "std::chrono::system_clock" and "std::chrono::steady_clock"?

`std::chrono::system_clock` and `std::chrono::steady_clock` are two different clocks provided by the C++ standard library for measuring and manipulating time. They serve different purposes and have distinct characteristics.

### std::chrono::system_clock

`std::chrono::system_clock` represents the system-wide real-time clock. It is designed to measure the current time according to the system's clock (wall-clock time). This is the clock you would typically use to get the current date and time.

#### Characteristics:
1. **System-wide Time**:
   - `system_clock` is tied to the time maintained by the operating system, often reflecting the local time or UTC, depending on the system configuration.
   
2. **Not Monotonic**:
   - The time reported by `system_clock` can be adjusted through various means such as NTP (Network Time Protocol) updates, manual changes by the user, or adjustments for daylight saving time.
   
3. **Suitable for Real-world Time**:
   - Useful for tasks involving real-world time measurement, such as logging timestamps, calculating durations between real-world events, or scheduling tasks at a specific date/time.

4. **Timezone Aware**:
   - You can convert `system_clock::time_point` to calendar time that represents real-world dates and times.

#### Example:
```cpp
#include <iostream>
#include <chrono>
#include <ctime>

int main() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::cout << "Current time: " << std::ctime(&now_c);
    return 0;
}
```

### std::chrono::steady_clock

`std::chrono::steady_clock` represents a monotonic clock that is guaranteed to be steady. This means that it will never go backwards and is not subject to system clock changes. It is designed for measuring time intervals and durations reliably.

#### Characteristics:
1. **Monotonic**:
   - `steady_clock` ensures that the time always increases monotonically, which is crucial for measuring durations and intervals accurately without being affected by system time updates or adjustments.

2. **Not Related to Real-world Time**:
   - The values returned by `steady_clock` do not have any direct relation to real-world date and time. They are meant for measuring elapsed time accurately.

3. **Suitable for Timers and Benchmarking**:
   - Ideal for timing operations, benchmarking code, and implementing timeouts where a consistent passage of time is required.

4. **Higher Guarantees of Precision**:
   - Since it is designed for state consistency, it provides higher guarantees about the precision and stability of measurements.

#### Example:
```cpp
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    auto start = std::chrono::steady_clock::now();
    
    // Simulate work
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    std::cout << "Elapsed time: " << duration << " seconds" << std::endl;
    return 0;
}
```

### Summary

- **`std::chrono::system_clock`**:
  - Represents system-wide real-time.
  - Not monotonic; can be adjusted.
  - Suitable for real-world time measurements and timestamps.
  - Can be converted to calendar time.

- **`std::chrono::steady_clock`**:
  - Represents a monotonic clock.
  - Always moves forward; not affected by system time changes.
  - Suitable for measuring elapsed time intervals, benchmarking, and precise timing.
  - Does not represent real-world time.

Choosing the right clock depends on your use case:
- Use `system_clock` when you need to work with current real-world time (e.g., logging, displaying the current date and time).
- Use `steady_clock` when you need reliable time intervals (e.g., measuring execution duration, implementing timeouts).

Q: how to system time tick count in c++?

In C++, you can retrieve the system time tick count using the `std::chrono` library, which provides a high-resolution clock and various utilities to measure time intervals. You can utilize `std::chrono::steady_clock` (or `std::chrono::high_resolution_clock` if very high resolution is needed, though its availability and characteristics are system-dependent) for this purpose.

Here's an example of how to get the system time tick count in C++:

```cpp
#include <iostream>
#include <chrono>

int main() {
    // Get the current time using steady_clock
    auto tick_count = std::chrono::steady_clock::now();

    // Print tick count in raw form
    std::cout << "Tick count (raw): " << tick_count.time_since_epoch().count() << std::endl;

    // For a more human-readable format, convert to duration in milliseconds
    auto tick_count_in_millis = std::chrono::duration_cast<std::chrono::milliseconds>(tick_count.time_since_epoch()).count();
    std::cout << "Tick count (milliseconds since epoch): " << tick_count_in_millis << " ms" << std::endl;

    return 0;
}
```

### Explanation

1. **Get the Current Time**:
   - `auto tick_count = std::chrono::steady_clock::now();` retrieves the current time point using `std::chrono::steady_clock`.

2. **Print the Raw Tick Count**:
   - `tick_count.time_since_epoch().count()` provides the raw tick count since the epoch (which may not necessarily be the Unix epoch, depending on the clock used).

3. **Convert to Milliseconds**:
   - `std::chrono::duration_cast<std::chrono::milliseconds>(tick_count.time_since_epoch()).count();` converts the tick count to milliseconds for a more human-readable format.

### Detailed Information About Clocks

- **std::chrono::steady_clock**: This clock is guaranteed to be steady, meaning it won't be adjusted once the system starts (good for measuring intervals). The tick count from `steady_clock` is useful for measuring elapsed time but does not correspond to a specific calendar time.

- **std::chrono::system_clock**: This clock can be adjusted (e.g., by NTP) and represents the wall time, which you can convert to calendar time. It might not be suitable for measuring intervals where monotonic behavior is required.

- **std::chrono::high_resolution_clock**: Potentially offers the highest available resolution, but its availability and characteristics can vary. Often, it is typedef'd to one of the other clocks (e.g., `steady_clock` or `system_clock`).

### Example Using `std::chrono::system_clock`

For completeness, if you want to check the tick count using `std::chrono::system_clock`:

```cpp
#include <iostream>
#include <chrono>

int main() {
    // Get the current time using system_clock
    auto tick_count = std::chrono::system_clock::now();

    // Print tick count in raw form
    std::cout << "Tick count (raw): " << tick_count.time_since_epoch().count() << std::endl;

    // For a more human-readable format, convert to duration in milliseconds
    auto tick_count_in_millis = std::chrono::duration_cast<std::chrono::milliseconds>(tick_count.time_since_epoch()).count();
    std::cout << "Tick count (milliseconds since epoch): " << tick_count_in_millis << " ms" << std::endl;

    return 0;
}
```

Selecting the appropriate clock and tick count format depends on your specific requirements, such as whether you need a steady, high-resolution, or calendar time-based measurement.

*/