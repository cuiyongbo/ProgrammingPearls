# Golang tutorial

* [Tutorial: Get started with Go](https://go.dev/doc/tutorial/getting-started)

* go-proxy config

```bash
go env -w GOPROXY="https://goproxy.cn,https://proxy.golang.org,direct"
go env -w GOPRIVATE="*.everphoto.cn,git.smartisan.com"
go env -w GOSUMDB="sum.golang.google.cn"
```

* project hierarchy

```bash
# tree tutorial/
tutorial/
├── greetings
│   ├── go.mod
│   ├── greetings.go
│   └── greetings_test.go
└── hello
    ├── go.mod
    └── hello.go
```