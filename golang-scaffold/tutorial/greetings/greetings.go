package greetings

import (
	"fmt"
	"errors"
	"math/rand"
)

/*
# intialize dependency tracking
go mod init example/greetings
*/

func Hello(name string) (string, error) {
	// no name was given, return an error with a message
	if name == "" {
		return "", errors.New("empty name")
	}
	// Return a greeting that embeds the name in a message
	message := fmt.Sprintf(randFormat(), name)
	return message, nil
}

func randFormat() string {
	greeting_formats := []string{
        "Hi, %v. Welcome!",
        "Great to see you, %v!",
        "Hail, %v! Well met!",
	}
	return greeting_formats[rand.Intn(len(greeting_formats))]
}

func Hellos(names []string) (map[string]string, error) {
	//  In Go, you initialize a map with the following syntax: make(map[key-type]value-type)
	messages := make(map[string]string)
	// In this for loop, range returns two values: the index of the current item in the loop and a copy of the item's value.
	// You don't need the index, so you use the Go blank identifier (an underscore) to ignore it
	for _, name := range names {
		message, err := Hello(name)
		if err != nil {
			return nil, err
		}
		messages[name] = message
	}
	return messages, nil
}