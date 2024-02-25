package greetings

/*
cuiyongbo@f10d9f434c5e:~/workspace/cyb_scaffold/greetings$ pwd
/home/cuiyongbo/workspace/cyb_scaffold/greetings

cuiyongbo@f10d9f434c5e:~/workspace/cyb_scaffold/greetings$ go test -v
=== RUN   TestHelloName
--- PASS: TestHelloName (0.00s)
=== RUN   TestHelloEmpty
--- PASS: TestHelloEmpty (0.00s)
PASS
ok      example.com/greetings   0.004s
*/

// Ending a file's name with _test.go tells the go test command that this file contains test functions.
import (
	"testing"
	"regexp"
)

/*
Test function names have the form TestName, where Name says something about the specific test.
Also, test functions take a pointer to the testing package's testing.T type as a parameter.
You use this parameter's methods for reporting and logging from your test.
*/

// TestHelloName calls greetings.Hello with a name, checking
// for a valid return value.
func TestHelloName(t *testing.T) {
    name := "Gladys"
    want := regexp.MustCompile(`\b`+name+`\b`)
    msg, err := Hello("Gladys")
    if !want.MatchString(msg) || err != nil {
        t.Fatalf(`Hello("Gladys") = %q, %v, want match for %#q, nil`, msg, err, want)
    }
}

// TestHelloEmpty calls greetings.Hello with an empty string,
// checking for an error.
func TestHelloEmpty(t *testing.T) {
    msg, err := Hello("")
    if msg != "" || err == nil {
        t.Fatalf(`Hello("") = %q, %v, want "", error`, msg, err)
    }
}