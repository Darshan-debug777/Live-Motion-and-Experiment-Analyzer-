import os
import shutil
from tqdm import tqdm

folder_path = input("Enter the path of folder to organize:")
file_types = {
    "Images": [".jpg", ".jpeg", ".png", ".gif"],
    "Documents": [".pdf", ".docx", ".txt"],
    "Videos": [".mp4", ".mov"],
    "Music": [".mp3", ".wav"],
    "Archives": [".zip", ".rar"],
}
files = os.listdir(folder_path)
for filename in tqdm(files, desc="Organizing files"):
    file_path = os.path.join(folder_path, filename)
    ext = os.path.splitext(filename)[1].lower()
    for category, extensions in file_types.items():
        if ext in extensions:
            category_folder = os.path.join(folder_path, category)
            os.makedirs(category_folder, exist_ok=True)
            shutil.move(file_path, category_folder)
            print(f"Moved: {filename} to {category}")
            break