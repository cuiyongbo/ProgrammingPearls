```
Q: how to declare th signature of a python function?

In Python, you can declare the signature of a function using type hints, which are part of the `typing` module. Type hints allow you to specify the expected types of the function's arguments and its return value. This can improve code readability and help with static type checking.

Here's how you can declare the signature of a Python function:

### Basic Function Signature

```python
def add(a: int, b: int) -> int:
    return a + b
```

### Explanation

- `a: int` and `b: int` specify that the function `add` expects two arguments, both of type `int`.
- `-> int` specifies that the function returns an `int`.

### Using the `typing` Module

For more complex types, you can use the `typing` module. Here are some examples:

#### Optional Arguments

```python
from typing import Optional

def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello, World!"
    return f"Hello, {name}!"
```

#### Lists and Dictionaries

```python
from typing import List, Dict

def process_data(data: List[int]) -> Dict[str, int]:
    return {"sum": sum(data), "max": max(data), "min": min(data)}
```

#### Tuples

```python
from typing import Tuple

def get_coordinates() -> Tuple[int, int]:
    return (10, 20)
```

#### Callable

```python
from typing import Callable

def execute_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)
```

### Example with Multiple Types

You can also specify multiple types using `Union`:

```python
from typing import Union

def parse_value(value: Union[int, str]) -> str:
    if isinstance(value, int):
        return f"Integer: {value}"
    return f"String: {value}"
```

### Example with Custom Classes

You can use type hints with custom classes as well:

```python
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

def celebrate_birthday(person: Person) -> None:
    person.age += 1
    print(f"Happy Birthday, {person.name}! You are now {person.age} years old.")
```

### Example with Generic Types

For functions that work with generic types, you can use `TypeVar`:

```python
from typing import TypeVar, List

T = TypeVar('T')

def get_first_element(elements: List[T]) -> T:
    return elements[0] if elements else None
```

### Conclusion

Type hints are a powerful feature in Python that can help you write more readable and maintainable code. They are especially useful in larger codebases and when working in teams, as they provide clear documentation of what types are expected and returned by functions.

### Full Example

Here is a complete example demonstrating various type hints:

```python
from typing import Optional, List, Dict, Tuple, Callable, Union, TypeVar

T = TypeVar('T')

def add(a: int, b: int) -> int:
    return a + b

def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello, World!"
    return f"Hello, {name}!"

def process_data(data: List[int]) -> Dict[str, int]:
    return {"sum": sum(data), "max": max(data), "min": min(data)}

def get_coordinates() -> Tuple[int, int]:
    return (10, 20)

def execute_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def parse_value(value: Union[int, str]) -> str:
    if isinstance(value, int):
        return f"Integer: {value}"
    return f"String: {value}"

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

def celebrate_birthday(person: Person) -> None:
    person.age += 1
    print(f"Happy Birthday, {person.name}! You are now {person.age} years old.")

def get_first_element(elements: List[T]) -> T:
    return elements[0] if elements else None
```

This example covers basic types, optional types, collections, custom classes, and generic types.

Q: my function return two values: list of Meta1, RunMeta, how to declare my function then?

To declare the signature of your function `foo` that returns a tuple containing a list of `Meta` objects and an instance of `RunMetadata`, you can use the `typing` module. Specifically, you will use `List` and `Tuple` to specify the types of the returned values.

Here is how you can declare the function signature:

```python
from typing import List, Tuple

class Meta:
    pass

class RunMetadata:
    pass

def foo() -> Tuple[List[Meta], RunMetadata]:
    a = []  # list of Meta
    m = RunMetadata()
    return a, m
```

### Explanation

- `List[Meta]` specifies that the first element of the returned tuple is a list of `Meta` objects.
- `RunMetadata` specifies that the second element of the returned tuple is an instance of `RunMetadata`.
- `Tuple[List[Meta], RunMetadata]` specifies that the function returns a tuple containing a list of `Meta` objects and an instance of `RunMetadata`.

### Full Example

Here is the complete code with the function signature:

```python
from typing import List, Tuple

class Meta:
    pass

class RunMetadata:
    pass

def foo() -> Tuple[List[Meta], RunMetadata]:
    a = []  # list of Meta
    m = RunMetadata()
    return a, m

# Example usage
meta_list, run_meta = foo()
print(meta_list)  # Output: []
print(run_meta)   # Output: <__main__.RunMetadata object at 0x...>
```

This way, the function signature clearly indicates what types of values the function returns, improving code readability and enabling better static type checking.
```