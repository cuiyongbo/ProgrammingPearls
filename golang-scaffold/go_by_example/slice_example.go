package main

import "fmt"

func slice_example() {
	fmt.Println("Running slice_example")

    // us []T to create a slice of type T
    var s []string
    fmt.Println("uninit:", s, s == nil, len(s) == 0)
   

    var failedIdx []int
    dataParsedStatus := []bool{true, false, true}
    for idx, v := range dataParsedStatus {
      if !v {
        failedIdx = append(failedIdx, idx)
      }
    }
    fmt.Println("dataParsedStatus:", dataParsedStatus, "failedIdx:", failedIdx)


	// To create an empty slice with non-zero length, use the builtin make.
	// if we know the slice is going to grow ahead of time, it’s possible to pass a capacity explicitly as an additional parameter to make
    s = make([]string, 3)
    fmt.Println("emp:", s, "len:", len(s), "cap:", cap(s))

    s[0] = "a"
    s[1] = "b"
    s[2] = "c"
    fmt.Printf("set: %v\n", s)
    fmt.Println("set:", s)
    fmt.Println("get:", s[2])

    fmt.Println("len:", len(s))

	// append returns a slice containing one or more new values. Note that we need to accept a return value from append as we may get a new slice value.
    s = append(s, "d")
    s = append(s, "e", "f")
    fmt.Println("apd:", s)

	// Slices can also be copy’d. Here we create an empty slice c of the same length as s and copy into c from s.
	// implement deep copy semantic
	// len returns the length of the slice as expected.
    c := make([]string, len(s))
    copy(c, s)
	c[0] = "x"
    fmt.Println("cpy:", c)
    fmt.Println("ori:", s)

    l := s[2:5]
    fmt.Println("sl1:", l)

    l = s[:5]
    fmt.Println("sl2:", l)

    l = s[2:]
    fmt.Println("sl3:", l)

	nums := []int{0,1,2,3,4,5,6,7,8,9}
	fmt.Printf("num[%d:%d]: %v\n", 0, 10, nums[0:10])
	fmt.Printf("num[%d:%d]: %v\n", 1, 5, nums[1:5])
	fmt.Printf("num[%d:%d]: %v\n", 3, 7, nums[3:7])

    t := []string{"g", "h", "i"}
    fmt.Println("dcl:", t)

    twoD := make([][]int, 3)
    for i := 0; i < 3; i++ {
        innerLen := i + 1
        twoD[i] = make([]int, innerLen)
        for j := 0; j < innerLen; j++ {
            twoD[i][j] = i + j
        }
    }
    fmt.Println("2d: ", twoD)
}