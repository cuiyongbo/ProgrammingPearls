package main

// https://gobyexample.com/

import (
  "fmt"
  "log"
)


func main() {
  log.SetPrefix("go_by_example: ")
  log.SetFlags(0)

  fmt.Println("start Running main...")

  var s1 []string
  fmt.Println(len(s1))
  s2 := make([]string, 0)
  fmt.Println(len(s2))

  //constants_example()
  //variables_example()
  //interfaces_example()
  //if_example()
  //switch_example()
  //struct_example()
  //array_example() // similar to c++ array
  //slice_example() // similar to c++ vector
  //map_example() // similar to c++ unordered_map
  //reflect_example()

  // decode and encode json
  //marshal_example()
  //unmarshal_example()
  //unmarshal_abitrary_data_example()
  //unmarshal_infinity()
  //unmarshal_infinity_2()

  fmt.Println("Finish Running main...")
}
