```
Q: how to use python dataclasses package?

The `dataclasses` package in Python provides a decorator and functions for automatically adding special methods to user-defined classes. This can simplify the creation of classes that are primarily used to store data. Below is a guide on how to use the `dataclasses` package.

### Basic Usage

To use the `dataclasses` package, you need to import it and use the `@dataclass` decorator on your class.

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
```

### Features

#### Automatic `__init__` Method

The `@dataclass` decorator automatically generates an `__init__` method for you:

```python
p = Person(name="John Doe", age=30)
print(p.name)  # Output: John Doe
print(p.age)   # Output: 30
```

#### Automatic `__repr__` Method

A `__repr__` method is also generated, which provides a string representation of the object:

```python
print(p)  # Output: Person(name='John Doe', age=30)
```

#### Automatic `__eq__` Method

The `__eq__` method is generated to allow comparison between instances:

```python
p1 = Person(name="John Doe", age=30)
p2 = Person(name="John Doe", age=30)
print(p1 == p2)  # Output: True
```

### Default Values

You can specify default values for fields:

```python
@dataclass
class Person:
    name: str
    age: int = 25

p = Person(name="Jane Doe")
print(p.age)  # Output: 25
```

### Field Metadata

You can add metadata to fields using the `field` function:

```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int = field(default=25, metadata={"unit": "years"})

p = Person(name="Jane Doe")
print(p.age)  # Output: 25
print(Person.__dataclass_fields__['age'].metadata)  # Output: {'unit': 'years'}
```

### Immutable Dataclasses

You can make dataclasses immutable by setting `frozen=True`:

```python
@dataclass(frozen=True)
class Person:
    name: str
    age: int

p = Person(name="John Doe", age=30)
# p.age = 31  # This will raise a FrozenInstanceError
```

### Post-Initialization Processing

You can define a `__post_init__` method for additional processing after the automatic `__init__` method:

```python
@dataclass
class Person:
    name: str
    age: int

    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

p = Person(name="John Doe", age=30)  # This works fine
# p = Person(name="John Doe", age=-5)  # This will raise a ValueError
```

### Example

Here is a complete example demonstrating various features:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Person:
    name: str
    age: int = field(default=25, metadata={"unit": "years"})
    email: str = field(default="unknown@example.com")

    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

p = Person(name="Jane Doe")
print(p)  # Output: Person(name='Jane Doe', age=25, email='unknown@example.com')
print(Person.__dataclass_fields__['age'].metadata)  # Output: {'unit': 'years'}
```

### Conclusion

The `dataclasses` package is a powerful tool for simplifying the creation of classes that are primarily used to store data. By using the `@dataclass` decorator, you can automatically generate common special methods and reduce boilerplate code.

Q: how to define a class with all fields optional decorated with dataclass

To define a class with all fields optional using the `@dataclass` decorator, you can use the `typing` module to specify that each field can be of type `Optional` (which is an alias for `Union[T, None]`). Additionally, you can provide a default value of `None` for each field.

Here's an example:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Person:
    name: Optional[str] = field(default=None)
    age: Optional[int] = field(default=None)
    email: Optional[str] = field(default=None)

# Example usage
p1 = Person()
print(p1)  # Output: Person(name=None, age=None, email=None)

p2 = Person(name="John Doe")
print(p2)  # Output: Person(name='John Doe', age=None, email=None)

p3 = Person(name="Jane Doe", age=30, email="jane.doe@example.com")
print(p3)  # Output: Person(name='Jane Doe', age=30, email='jane.doe@example.com')
```

### Explanation

1. **Importing Required Modules**:
   - `dataclass` and `field` from the `dataclasses` module.
   - `Optional` from the `typing` module.

2. **Defining the Class**:
   - Use the `@dataclass` decorator to automatically generate special methods like `__init__`, `__repr__`, and `__eq__`.
   - Define each field as `Optional` and provide a default value of `None` using `field(default=None)`.

### Example with Default Values

If you want to provide specific default values instead of `None`, you can do so directly:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Person:
    name: Optional[str] = field(default="Unknown")
    age: Optional[int] = field(default=0)
    email: Optional[str] = field(default="unknown@example.com")

# Example usage
p1 = Person()
print(p1)  # Output: Person(name='Unknown', age=0, email='unknown@example.com')

p2 = Person(name="John Doe")
print(p2)  # Output: Person(name='John Doe', age=0, email='unknown@example.com')

p3 = Person(name="Jane Doe", age=30, email="jane.doe@example.com")
print(p3)  # Output: Person(name='Jane Doe', age=30, email='jane.doe@example.com')
```

In this example, the fields `name`, `age`, and `email` have default values of `"Unknown"`, `0`, and `"unknown@example.com"`, respectively. This way, if no value is provided for a field, it will use the specified default value instead of `None`.

```
