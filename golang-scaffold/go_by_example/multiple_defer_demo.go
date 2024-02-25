package main

import "fmt"

/*
in Go, you can have multiple defer statements in a single function. All deferred calls are pushed onto a stack, and when the function returns, the deferred calls are executed in last-in, first-out (LIFO) order. This means the last defer statement in the function will be the first one to be executed when the function returns.
*/

func multiDeferExample() {
	defer fmt.Println("First deferred call")
	defer fmt.Println("Second deferred call")
	defer fmt.Println("Third deferred call")

	fmt.Println("Function logic executed")
}

func main() {
	multiDeferExample()
}