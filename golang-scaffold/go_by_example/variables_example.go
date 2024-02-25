package main

import "fmt"

func variables_example() {
	fmt.Println("Runing variables_example")

	// var declares 1 or more variables.
    var a = "initial"
    fmt.Println(a)

    var b, c int = 1, 2
    fmt.Println(b, c)

    var d = true
    fmt.Println(d)

	// Variables declared without a corresponding initialization are zero-valued. For example, the zero value for an int is 0.
    var e int
    fmt.Println(e)

	// The := syntax is shorthand for declaring and initializing a variable. This syntax is only available inside functions.
    f := "apple"
    fmt.Println(f)
}
