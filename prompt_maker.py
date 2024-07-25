import os


def format_py_files():
    # Get all .py files in the current directory and subdirectories
    py_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))

    # Create or overwrite the output file
    with open('formatted_py_files.txt', 'w') as output_file:
        for py_file in py_files:
            # Write the filename with its relative path
            output_file.write(f"{py_file}\n")

            # Read and write the contents of the .py file
            with open(py_file, 'r') as file:
                output_file.write(file.read())

            # Add a newline between files for better readability
            output_file.write('\n\n')

    print("Formatting complete. Check 'formatted_py_files.txt' for the result.")


if __name__ == "__main__":
    format_py_files()