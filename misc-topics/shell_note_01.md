```
Q: in shell script, what is the difference between "command1 && command2" and "command1; command2"

In shell scripting, the difference between `command1 && command2` and `command1; command2` lies in how the commands are executed and under what conditions they proceed.

### `command1 && command2`

The `&&` operator is a logical AND operator. It means that `command2` will only be executed if `command1` succeeds (i.e., `command1` exits with a status code of 0).

**Syntax:**
```sh
command1 && command2
```

**Behavior:**
- `command1` is executed.
- If `command1` succeeds (exits with a status code of 0), then `command2` is executed.
- If `command1` fails (exits with a non-zero status code), then `command2` is not executed.

**Example:**
```sh
mkdir new_directory && cd new_directory
```
- In this example, `cd new_directory` will only be executed if `mkdir new_directory` succeeds (i.e., the directory is created successfully). If `mkdir` fails (e.g., because the directory already exists), `cd new_directory` will not be executed.

### `command1; command2`

The `;` operator is a command separator. It means that `command2` will be executed after `command1` regardless of whether `command1` succeeds or fails.

**Syntax:**
```sh
command1; command2
```

**Behavior:**
- `command1` is executed.
- `command2` is executed regardless of the exit status of `command1`.

**Example:**
```sh
echo "Hello"; echo "World"
```
- In this example, both `echo "Hello"` and `echo "World"` will be executed independently. The execution of the second `echo` command does not depend on the success or failure of the first `echo` command.

### Comparison Table

| Syntax                | Behavior                                                                                   |
|-----------------------|--------------------------------------------------------------------------------------------|
| `command1 && command2`| `command2` is executed only if `command1` succeeds (exits with a status code of 0).        |
| `command1; command2`  | `command2` is executed regardless of whether `command1` succeeds or fails.                 |

### Use Cases

- **`command1 && command2`**:
  - Use this when the execution of `command2` depends on the successful completion of `command1`.
  - Example: `compile_program && run_tests`
    - `run_tests` should only be executed if `compile_program` succeeds.

- **`command1; command2`**:
  - Use this when you want to execute `command2` regardless of the outcome of `command1`.
  - Example: `echo "Start"; run_process; echo "End"`
    - Both the `echo` statements should be executed regardless of whether `run_process` succeeds or fails.

### Practical Examples

1. **Using `&&`**:
    ```sh
    file_path="/path/to/file"
    [ -f "$file_path" ] && echo "File exists" && cat "$file_path"
    ```
    - This will check if the file exists and, if it does, print "File exists" and display the contents of the file.

2. **Using `;`**:
    ```sh
    echo "Cleaning up..."; rm -rf /some/directory; echo "Cleanup complete"
    ```
    - This sequence ensures that all echo statements are displayed regardless of whether the `rm` command succeeds or fails.

Understanding the difference between `&&` and `;` in shell scripting helps in writing more precise and controlled scripts for various scenarios.
```