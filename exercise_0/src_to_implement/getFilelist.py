import os.path


def getFilelist(folder):
    # Get the list of files inside a directories

    # list to store files
    file_dir = []
    dir_data = os.listdir(folder)
    dir_data = sorted(dir_data, key=lambda x: int(os.path.splitext(x)[0]))

    # Iterate directory
    for path in dir_data:
        # check if current path is a file
        if os.path.isfile(os.path.join(folder, path)):
            file_dir.append(folder + r"\\" + path)
    return file_dir


files = getFilelist(r"C:\Users\jkavy\DeepLearning\exercise0\src_to_implement\data\exercise_data")
print(files)