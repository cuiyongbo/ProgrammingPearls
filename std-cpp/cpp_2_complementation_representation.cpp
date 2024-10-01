#include <iostream>
#include <bitset>

template<typename T>
void printTwoComplementRepresentation(T val) {
    const int bit_width = sizeof(T) * 8;
    std::bitset<bit_width> bs(val);
    printf("printTwoComplementRepresentation(%d): %s\n", val, bs.to_string().c_str());
    //std::cout << "2's complement representation of " << val << " is: " << bs << std::endl;
}

void int32ComplementRepresentation(int32_t val) {
    const int bit_width = 32;
    std::string repr(bit_width, '0');
    bool is_negative = val < 0;
    int32_t dividend = is_negative ? -val : val;
    int32_t divisor = 2;
    int32_t quotient = 0;
    int32_t remainder = 0;
    int i = 0;
    do {
        i++;
        quotient = dividend/divisor;
        remainder = dividend%divisor;
        dividend = quotient;
        if (remainder) {
            repr[32-i] = '1';
        } else {
            repr[32-i] = '0';
        }
    } while (quotient != 0);

    if (is_negative) {
        //printf("1: %s\n", repr.c_str());
        // reverse every bit
        for (int i=0; i<repr.size(); i++) {
            if (repr[i] == '0') {
                repr[i] = '1';
            } else {
                repr[i] = '0';
            }
        }
        //printf("2: %s\n", repr.c_str());
        // +1
        int carry = 1; // initial 1
        for (int i=repr.size()-1; i>=0; i--) {
            int sum = (repr[i]-'0') + carry;
            //printf("iter %02d, repr[%02d]: %d, carry: %d, sum: %d\n", i, i, repr[i]-'0', carry, sum);
            if (sum > 1) {
                repr[i] = '0';
                carry = 1;
            } else {
                repr[i] = sum + '0';
                carry = 0;
            }
        }
        //printf("3: %s\n", repr.c_str());
    }
    printf("   int32ComplementRepresentation(%d): %s\n", val, repr.c_str());
}

int main() {
    // https://www.reddit.com/r/cpp_questions/comments/z27gw2/help_needed_with_understanding_whileloop_with/
    int num = 0;
    std::cout << "Enter an integer: ";
    while (std::cin >> num) {
        printTwoComplementRepresentation(num);
        int32ComplementRepresentation(num);
        std::cout << "Enter an integer: ";
    }
    return 0;
}

/*
Q: how to print 2's complement representation of an int in C++?

Printing the 2's complement representation of an integer in C++ involves converting the integer to its binary form and then displaying it. The 2's complement representation is a way of encoding signed integers in binary, where the most significant bit (MSB) indicates the sign (0 for positive, 1 for negative).

### Steps to Calculate 2's Complement Manually

1. **Convert the Positive Value to Binary**: Start with the absolute value of the negative integer and convert it to binary.
2. **Invert the Bits**: Flip all the bits (change 0s to 1s and 1s to 0s).
3. **Add 1**: Add 1 to the inverted bits to get the 2's complement representation.

### Example: Calculate 2's Complement of -5

1. **Convert 5 to Binary**: 
   - For an 8-bit representation: `00000101`
2. **Invert the Bits**: 
   - `11111010`
3. **Add 1**: 
   - `11111010` + `1` = `11111011`

So, the 2's complement representation of -5 in an 8-bit system is `11111011`.

### Steps to Print 2's Complement Representation

1. **Determine the Bit Width**: Decide the number of bits you want to use for the representation (e.g., 8, 16, 32, 64 bits).
2. **Convert to Binary**: Convert the integer to its binary form, taking into account the bit width.
3. **Handle Negative Numbers**: For negative numbers, ensure the binary representation correctly reflects the 2's complement form.

### Example Code

Here's a complete example of how to print the 2's complement representation of an integer in C++:

#### Step 1: Include Required Headers

```cpp
#include <iostream>
#include <bitset>
```

#### Step 2: Define a Function to Print 2's Complement Representation

```cpp
template <typename T>
void printTwosComplement(T value) {
    // Determine the bit width of the type
    constexpr size_t bitWidth = sizeof(T) * 8;

    // Convert the value to a bitset
    std::bitset<bitWidth> bits(value);

    // Print the bitset
    std::cout << "2's complement representation of " << value << " is: " << bits << std::endl;
}
```

#### Step 3: Use the Function to Print 2's Complement Representation

```cpp
int main() {
    int8_t value8 = -5;
    int16_t value16 = -5;
    int32_t value32 = -5;
    int64_t value64 = -5;

    printTwosComplement(value8);
    printTwosComplement(value16);
    printTwosComplement(value32);
    printTwosComplement(value64);

    return 0;
}
```

### Explanation

1. **Include Headers**: The `<bitset>` header is included to use the `std::bitset` class, which provides a convenient way to handle binary representations.
2. **Template Function**: The `printTwosComplement` function is defined as a template to handle different integer types (`int8_t`, `int16_t`, `int32_t`, `int64_t`).
3. **Bit Width**: The bit width of the type is determined using `sizeof(T) * 8`.
4. **Convert to Bitset**: The integer value is converted to a `std::bitset` of the appropriate bit width.
5. **Print Bitset**: The `std::bitset` is printed, showing the 2's complement representation of the integer.

### Example Output

```plaintext
2's complement representation of -5 is: 11111011
2's complement representation of -5 is: 1111111111111011
2's complement representation of -5 is: 11111111111111111111111111111011
2's complement representation of -5 is: 1111111111111111111111111111111111111111111111111111111111111011
```

### Summary

- **Bit Width**: Determine the bit width of the integer type.
- **Convert to Bitset**: Use `std::bitset` to convert the integer to its binary form.
- **Print Bitset**: Print the `std::bitset` to display the 2's complement representation.

By following these steps, you can print the 2's complement representation of an integer in C++ for various integer types. This approach ensures that the binary representation correctly reflects the 2's complement form, including handling negative numbers appropriately.
*/
