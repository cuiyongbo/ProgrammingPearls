package main

import "fmt"

func map_example() {
	fmt.Println("Running map_example")

	// To create an empty map, use the builtin make: make(map[key-type]val-type).
    m := make(map[string]int)
	// Set key/value pairs using typical name[key] = val syntax.
    m["k1"] = 7
    m["k2"] = 13
		for k, v := range m {
			fmt.Printf("key: %v, value: %v", k, v)
		}

    v1 := m["k1"]
    fmt.Println("v1:", v1)
	// If the key doesn’t exist, the zero value of the value type is returned.
    v3 := m["k3"]
    fmt.Println("v3:", v3)
    fmt.Println("len:", len(m))

    delete(m, "k2")
    fmt.Println("map:", m)

	// The optional second return value when getting a value from a map indicates if the key was present in the map.
	// This can be used to disambiguate between missing keys and keys with zero values like 0 or "". 
	// Here we didn’t need the value itself, so we ignored it with the blank identifier _.
    _, prs2 := m["k2"]
    fmt.Println("k2 prs2:", prs2)
    _, prs1 := m["k1"]
    fmt.Println("k1 prs1:", prs1)

	testMap := map[string]interface{} {
		"k1": 1,
		"k2": 2,
		"op": "+",
	}
	if _, ok := testMap["x"]; ok {
		fmt.Printf("%v\n", testMap["x"])
	} else {
		fmt.Printf("x does not exist in map\n")
	}
	if _, ok := testMap["x"]; ok {
		fmt.Printf("%v\n", testMap["x"])
	} else {
		fmt.Printf("x does not exist in map\n")
	}
}