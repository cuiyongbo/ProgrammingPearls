package main

import (
	"fmt"
	"log"
	"time"
	//"unicode"
	//"strconv"
	"encoding/json"
	//"reflect"

)


// func Marshal(v interface{}) ([]byte, error)
// func Unmarshal(data []byte, v interface{}) error


type Message struct {
    Name string
    Body string
    Time time.Time
}

func marshal_example() {
	fmt.Println("Running marshal_example")
	m := Message{Name: "question", Body: "what is vector db for", Time: time.Now()}
	json_repr, err := json.Marshal(m)
	if err != nil {
		log.Printf("failed to json.Marshal, error: %v\n", err)
		return
	}
	fmt.Println("succeeded in json.Marshal, json: ", string(json_repr))
}

func unmarshal_example() {
	fmt.Println("Running unmarshal_example")
	b := []byte(`{"Name":"question","Body":"what is vector db for","Time":"2023-08-03T17:04:35.249352+08:00"}`)
	var m Message
	err := json.Unmarshal(b, &m)
	if err != nil {
		log.Printf("failed to json.Unmarshal, error: %v\n", err)
		return
	}
	fmt.Printf("succeeded in json.Unmarshal, result: %#v\n", m)
}

func unmarshal_abitrary_data_example() {
	fmt.Println("Running unmarshal_abitrary_data_example")
	b := []byte(`{"Name":"Wednesday","Age":6,"Parents":["Gomez","Morticia"]}`)
	var p map[string]interface{}
	err := json.Unmarshal(b, &p)
	if err != nil {
		log.Printf("failed to json.Unmarshal, error: %v\n", err)
		return
	}
	for k, v := range p {
		// report error: invalid syntax tree: use of .(type) outside type switch
		//fmt.Printf("key: %s, value_type: %s, value: %v", k, v.(type), v)
		switch vv := v.(type) {
		case string:
			fmt.Printf("key: %s, string value: %s\n", k, vv)
		case float64:
			fmt.Printf("key: %s, float64 value: %f\n", k, vv)
		case []interface{}:
			fmt.Printf("key: %s, array value: \n", k)
			for i, u := range vv {
				fmt.Printf("\t%d: %v\n", i, u)
			}
		default:
			fmt.Printf("value of %s is unexpected\n", k)
		}
	}
}
