# Python Style

Production code should be performant, easy to maintain, and easy to debug.
Coding style and standards are a critical and often overlooked aspect of software development.

Let's consider two examples:

1. Consistent naming conventions (e.g. `my_func` or `MyClass`). This enables developers
to have higher velocity when switching between projects and leads to less bugs.

2. Using an ill-advised mutable default value as an argument to a function:

```python
# ill-advised mutable default value

def append(num, num_lst=[]):
    num_lst.append(number)
    return num_lst

append(1)  # expected: [1], result: [1]
append(2)  # expected: [2], result: [1, 2]
append(3)  # expected: [3], result: [1, 2, 3]

# advised use None

def append(number, num_lst=None):
    if num_lst is None:
        num_lst = []
    num_lst.append(number)
    return num_lst

append(1)  # expected: [1], result: [1]
append(2)  # expected: [2], result: [2]
append(3)  # expected: [3], result: [3]
```

The [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) is
a reference that includes an exhaustive list of python best practices that can be used
to improve the quality of production code. Incorporation of even a few new best practices
starts a ripple effect as you write and review merge requests.
