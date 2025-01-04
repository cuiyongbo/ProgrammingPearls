# How to use the traditional `vi` editor

If you are offline and need to use the traditional `vi` editor, here are some tips on how to interact with it using basic commands. The `^` characters you see when using the arrow keys in `vi` are typically control characters that indicate the terminal isn't interpreting the keys as expected. Instead of using arrow keys, you can use `vi` navigation commands.

### Basic `vi` Commands

Here are some important `vi` commands to get you started:

#### Entering and Exiting Modes

- **Normal Mode**: This is the default mode when you open `vi`. You navigate and control the text here.
- **Insert Mode**: Press `i` to enter Insert Mode from Normal Mode. You can type to insert text.
- **Command Mode**: Press `:` to enter Command Mode from Normal Mode for file operations like save and quit.

#### Navigation

- `h`: Move left
- `j`: Move down
- `k`: Move up
- `l`: Move right

These keys are used in Normal Mode to navigate within the text file.

#### Edit and Save

- `i`: Enter Insert Mode before the cursor
- `a`: Enter Insert Mode after the cursor
- `x`: Delete the character under the cursor- `dd`: Delete the current line
- `:w`: Save the file
- `:q`: Quit `vi`
- `:wq`: Save and quit `vi`
- `:q!`: Quit without saving

#### Search

- `/pattern`: Search for `pattern` in the text. Press `n` to go to the next occurrence and `N` to go to the previous one.

### Example Workflow

1. **Open a file**:

   ```bash
   vi 1.txt
   ```

2. **Enter Insert Mode**:
   - Press `i` to begin inserting text at the current cursor position.

3. **Type your text**:
   - Type normally; the text will be inserted where the cursor is.

4. **Return to Normal Mode**:
   - Press `Esc` to return to Normal Mode from Insert Mode.

5. **Navigate the text**:
   - Use `h`, `j`, `k`, and `l` to move the cursor around.

6. **Make changes**:
   - Use `x` to delete characters, `dd` to delete lines, etc.

7. **Save your changes**:
   - Press `:` to enter Command Mode.
   - Type `w` and press `Enter` to save the file.
   
8. **Quit `vi`**:
   - Press `:` to enter Command Mode.
   - Type `q` and press `Enter` to quit `vi`.

### Example Session

Here's an example of a typical `vi` session:

1. Open `vi`:
   ```bash
   vi 1.txt
   ```

2. Enter Insert Mode and add text:
   ```
   iHello, this is a text file.
   [Press Esc to exit Insert Mode]
   ```

3. Navigate to the end of the line and add more text:
   ```
   A new text at the end.
   [Press Esc to exit Insert Mode]
   ```

4. Delete a character and a line:
   ```
   x  (delete character under cursor)
   dd (delete current line)
   ```

5. Save and exit:
   ```
   :wq [Press Enter]
   ```

By using the traditional `vi` editor commands, you can effectively edit text even when you are offline and without enhanced features provided by `vim`.
