#include <stdio.h>
#include <stdlib.h>

#define ARRAY_LENGTH(a) (sizeof(a)/sizeof(a[0]))

struct data 
{
    int nr;
    char const *value;
} dat[] = 
{
    {1, "Foo"}, {2, "Bar"}, {3, "Hello"}, {4, "World"}
};

int data_cmp(void const *lhs, void const *rhs)
{
    struct data* l = lhs;
    struct data* r = rhs;
    return (l->nr > r->nr) - (l->nr < r->nr); 
}

int main(void)
{
    struct data key = { .nr = 3 };
    struct data const *res = bsearch(&key, dat, ARRAY_LENGTH(dat), sizeof(dat[0]), data_cmp);
    if (res) {
        printf("No %d: %s\n", res->nr, res->value);
    } else {
        printf("No %d not found\n", key.nr);
    }
}
