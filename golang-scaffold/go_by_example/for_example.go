package main

import "fmt"

func main() {
  for j := 7; j <= 9; j++ {
    fmt.Println(j)
  }

  for {
    fmt.Println("loop")
    break
  }

  var ShardCountInt int64
  ShardCountInt = 10
  for i:=int64(0); i<m.ShardCountInt; i++ {
    if i%2 == 0 {
      continue
    }
    fmt.Println(i)
  }

  batchSize := 15
  keys := make([]int, 0)
  for j := 0; j <= 100; j++ {
    keys = append(keys, j)
  }
  // Split keys into batches
  var batches [][]int
  for i := 0; i < len(keys); i += batchSize {
    end := i + batchSize
    if end > len(keys) {
      end = len(keys)
    }
    batches = append(batches, keys[i:end])
  }
}