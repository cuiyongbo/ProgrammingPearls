package main

import (
	"fmt"
	"reflect"
	"regexp"
)

func main() {
	inputString := `time.total=45.00 time.scan=44.00 count.index=10.00 count.scan=10.00 count.context_id=1.00 count.scan_storage_size=19921.00`
	
	re := regexp.MustCompile(`count.scan_storage_size=(\d+\.\d+)`)
	matches := re.FindStringSubmatch(inputString)

	if len(matches) > 1 {
		fmt.Printf("Extracted value: %v, type: %v\n", matches[1], reflect.TypeOf(matches[1]))
	} else {
		fmt.Println("Pattern not found")
	}
}

I have struct definitions as following, and when I update `ShardCountInt`, I got this error: cannot assign to struct field collectionConf.Models[k].ShardCountInt in map

var collectionConf CollectionConfDetail
// fill collectionConf
for k, _ := range collectionConf.Models {
	// calculate ShardCountInt
	collectionConf.Models[k].ShardCountInt = 9
}

// 解析 collection 配置, 从中获取 CpuQuota
type ModelDetail struct {
  ModelMeta struct {
    Corpus  string `json:"corpus,required"`
    ModelName  string `json:"model_name,required"`
    Version  string `json:"version,required"`
    Opname  string `json:"opname,required"`
    CpuQuota float64  `json:"CPUQuota,required"`
  } `json:"model_meta,required"`
  ShardCount string  `json:"shard_count,required"`  // 注意元信息保存的类型是 string
  ShardCountInt int64
}

type CollectionConfDetail struct {
  Models  map[string]ModelDetail `json:"models"`
  ColumnConf struct {
    CollectionName string     `json:"collection_name,required"`
    AccountID       int64     `json:"account_id"`
    UserID          int64     `json:"user_id"`
    UserName        string    `json:"user_name"`
    InstanceNO      string    `json:"instance_no"`
  } `json:"column_conf,required"`
}