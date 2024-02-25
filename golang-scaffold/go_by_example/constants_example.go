package main

import (
    "fmt"
    "math"
)

// const declares a constant value.
// A const statement can appear anywhere a var statement can.
const s string = "constant"

func constants_example() {
	fmt.Println("Runnig constants_example")
    fmt.Println(s)
    const n = 500000000
    const d = 3e20 / n
    fmt.Println(d)
    fmt.Println(int64(d))
	// A number can be given a type by using it in a context that requires one, such as a variable assignment or function call.
	// For example, here math.Sin expects a float64.
	// simlilar to implict convertion in c++
    fmt.Println(math.Sin(n))
}