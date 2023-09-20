import os

def write_python_file_contents(directory):
    with open("total_code.txt", "w") as total_code_file:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    total_code_file.write(f"filename: {relative_path}\ncontents:\n")

                    with open(file_path, "r") as python_file:
                        total_code_file.write(python_file.read())
                    total_code_file.write("\n")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    write_python_file_contents(directory)
