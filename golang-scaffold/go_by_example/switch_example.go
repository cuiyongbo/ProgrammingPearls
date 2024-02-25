package main

import (
    "fmt"
    "time"
)

func switch_example() {
	fmt.Println("Running switch_example")

    i := 2
    fmt.Print("Write ", i, " as ")
    switch i {
    case 1:
        fmt.Println("one")
    case 2:
        fmt.Println("two")
    case 3:
        fmt.Println("three")
    }

	switch i {
	case i%2:
		fmt.Printf("%d is an odd number\n", i)
	default:
		fmt.Printf("%d is an even number\n", i)
	}

    switch time.Now().Weekday() {
	// You can use commas to separate multiple expressions in the same case statement.
    case time.Saturday, time.Sunday:
        fmt.Println("It's the weekend")
    default:
        fmt.Println("It's a weekday")
    }

    t := time.Now()
    switch {
    case t.Hour() < 12:
        fmt.Println("It's before noon")
    default:
        fmt.Println("It's after noon")
    }

	// A type switch compares types instead of values. You can use this to discover the type of an interface value.
    whatAmI := func(i interface{}) {
        switch t := i.(type) {
        case bool:
            fmt.Println("I'm a bool")
        case int:
            fmt.Println("I'm an int")
        default:
            fmt.Printf("Don't know type %T\n", t)
        }
    }
    whatAmI(true)
    whatAmI(1)
    whatAmI("hey")
	whatAmI([]int{1, 2})
	whatAmI([2]int{1, 2})
}