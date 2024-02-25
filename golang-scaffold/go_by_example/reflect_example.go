package main

import (
  "fmt"
  "reflect"
)

func reflect_example() {
  fmt.Println("Running reflect_example")
  var num int = 42
  var str string = "Hello, World!"
  var arr []int = []int{1, 2, 3}
  var mp map[string]int = map[string]int{"apple": 1, "banana": 2}
  var non_sense interface{}
  var non_sense_list []interface{}

  fmt.Println(reflect.TypeOf(num)) // 输出: int
  fmt.Println(reflect.TypeOf(str)) // 输出: string
  fmt.Println(reflect.TypeOf(arr)) // 输出: []int
  fmt.Println(reflect.TypeOf(mp))  // 输出: map[string]int
  fmt.Println(reflect.TypeOf(non_sense))
  fmt.Println(reflect.TypeOf(non_sense_list))
}