def split_file(spam_train_txt, train_txt, validation_txt, split_line):
    try:
        with open(spam_train_txt, 'r') as file:
            lines = file.readlines()

        with open(train_txt, 'w') as file1:
            file1.writelines(lines[:split_line])

        with open(validation_txt, 'w') as file2:
            file2.writelines(lines[split_line:])

        print(f"File has been successfully split into {train_txt} and {validation_txt}.")
    
    except FileNotFoundError:
        print(f"The file {spam_train_txt} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
