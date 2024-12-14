
package main

import (
	"crypto/rand"
	"fmt"
	"log"
)

// RandomString generates a random string of length n using characters from a given set
func RandomString(n int) (string, error) {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	bytes := make([]byte, n) // Create a byte slice of the desired length
	if _, err := rand.Read(bytes); err != nil {
		return "", err // Return an error if there is one
	}
	for i, b := range bytes {
		bytes[i] = charset[b%byte(len(charset))] // Map each byte to a character in the charset
	}
	return string(bytes), nil // Convert the bytes to string and return
}

func main() {
	// Generate a random 10-character string
	randomString, err := RandomString(100)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(randomString, len(randomString))
}


