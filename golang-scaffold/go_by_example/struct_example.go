package main

import "fmt"

type person struct {
	name string
	age int
}

// You can safely return a pointer to local variable as a local variable will survive the scope of the function.
func newPerson(name string) *person {
	// You can name the fields when initializing a struct.
	// Omitted fields will be zero-valued.
	p := person{name: name}
	// Access struct fields with a dot.
	p.age = 42
	// An & prefix yields a pointer to the struct.
	return &p
}

func struct_example() {
	fmt.Println("Running struct_example")

	s := person{"Bob", 20}
	fmt.Println(s)
	sp := &s
	// You can also use dots with struct pointers - the pointers are automatically dereferenced.
	fmt.Println(sp.age)
	sp.age = 30
	fmt.Println(s)

	fmt.Println(&person{name: "Ann"})
	fmt.Println(newPerson("Tom"))

	// If a struct type is only used for a single value, we don’t have to give it a name. The value can have an anonymous struct type.
    dog := struct {
        name   string
        isGood bool
    }{
        "Rex",
        true,
    }
    fmt.Println(dog)

}