package main

import (
	"fmt"
	"math"
)

type geometry interface {
	area() float64
	perim() float64
}

type rect struct {
	width, height float64
}

func (r rect) area() float64 {
	return r.width * r.height
}

func (r rect) perim() float64 {
	return 2 * (r.width + r.height)
}


type circle struct {
	radius float64
}

func (c circle) area() float64 {
	return math.Pi * c.radius * c.radius
}

func (c circle) perim() float64 {
	return 2 * math.Pi * c.radius
}

func measure(g geometry) {
	fmt.Printf("geometry object: %#v\n", g)
	fmt.Printf("area: %f\n", g.area())
	fmt.Printf("perimeter: %f\n", g.perim())
}

func interfaces_example() {
	fmt.Println("Running interfaces_example")
	r := rect{width: 3, height: 4}
	measure(r)
	c := circle{radius: 5}
	measure(c)
}
