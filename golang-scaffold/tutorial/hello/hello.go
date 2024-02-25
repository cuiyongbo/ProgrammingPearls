package main

import (
    "fmt"
    "log"
    //"rsc.io/quote"

    "example.com/greetings"
)

/*
# intialize dependency tracking
go mod init example/hello
# TEST ONLY!! The command specifies that example.com/greetings should be replaced with ../greetings for the purpose of locating the dependency
go mod edit -replace example.com/greetings=../greetings
# auto fix depencies
go mod tidy
# run the program
go run .
*/

// function definition
func main() {
    /*
    Set properties of the predefined Logger, including
    the log entry prefix and a flag to disable printing
    the time, source file, and line number.
    */
    log.SetPrefix("greetings: ")
    log.SetFlags(0)

/*
    // hello world
    fmt.Println("Hello, World!")

    // call functions from an external module
    // don't import a module unless you would use it, otherwise `"rsc.io/quote" imported and not used`
    fmt.Println("quote.Go():", quote.Go())
    fmt.Println("quote.Glass(): ", quote.Glass())
    fmt.Printf("quote.Hello(): %s\n", quote.Hello())
    fmt.Printf("quote.Opt(): %s\n", quote.Opt())
*/

    // error handling
    // similar to a public member function in c++ class
    // In Go, the := operator is a shortcut for declaring and initializing a variable in one line (Go uses the value on the right to determine the variable's type).
    message, err := greetings.Hello("SONY")
    if err != nil {
        log.Fatal(err)
    } else {
        fmt.Printf("greetings.Hello: %s\n", message)
    }

    // ./hello.go:41:15: undefined: greetings.randFormat
    // similar to a private member function in c++ class
    // greetings.randFormat()

    // A slice of names. similar to vector type in c++
    names := []string{"Gladys", "Samantha", "Darrin"}
    messages, err := greetings.Hellos(names)
    if err != nil {
        log.Fatal(err)
    }

    //fmt.Println(messages)

    // loop through a map
    for key, value := range messages {
        fmt.Printf("%s: %s\n", key, value)
    }

}