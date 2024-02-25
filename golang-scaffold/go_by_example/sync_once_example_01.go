package main

import (
	"fmt"
	"sync"
)

/*
The sync.Once type provides a mechanism to guarantee that a specific function, referred to as the "once function," is executed only once throughout the lifetime of the program. This is typically used for one-time initialization tasks such as setting up global variables, creating caches, or establishing connections to external resources.

The sync.Once type has a single method called Do, which accepts a function as its argument. When the Do method is called, it checks whether the specified function has already been executed. If it hasn't, the function is executed exactly once, and subsequent calls to Do will not execute the function again.
*/

var (
	initialized bool
	initializeOnce sync.Once
)

func initialize() {
	fmt.Println("Initializing...")
	// Perform initialization tasks here.
	initialized = true
}

func main() {
	// Call initialize() multiple times, but it will only execute once.
	initializeOnce.Do(initialize)
	initializeOnce.Do(initialize)
	initializeOnce.Do(initialize)
	
	fmt.Println("Initialized:", initialized)
}