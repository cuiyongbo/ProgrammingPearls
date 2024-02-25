package main

import (
	"fmt"
	"sync"
)

var initializeOnce sync.Once

func initialize() {
	// Be catious, you have to catch the panic in the same goroutine where it happens
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("Panic recovered:", r)
		}
	}()
	fmt.Println("Initializing...")
	panic("Oops, something went wrong!")
}

func main() {
	initializeOnce.Do(initialize)
	initializeOnce.Do(initialize)
	fmt.Println("Program continues execution after panic.")
}
