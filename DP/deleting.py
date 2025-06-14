# Function to count images in a directory
import os
import shutil
import random


def count_images(directory):
    image_count = len([file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return image_count

# Function to delete folders with fewer than 30 images
def delete_folders(path):
    for root, dirs, files in os.walk(path):
        print("root:", root, "dirs:",dirs, "files:",files[0])
        break
#         for folder in dirs:
#             folder_path = os.path.join(root, folder)
#             image_count = count_images(folder_path)

#             if image_count < 140:
#                 print(f"Folder: {folder_path} - Image count: {image_count}")
#                 try:
#                     shutil.rmtree(folder_path)
#                     print(f"Deleted folder: {folder}")
#                 except OSError as e:
#                     print(f"Error: {e}")

# Delete subfolders with fewer than 30 images in each directory
for subdir, _, _ in os.walk("1"):
    delete_folders(r"train")


# def delete_extra_images(root_folder):
#     subdir_images = {}
#     min_images = float('inf')  # Set initial minimum count to infinity

#     # Loop through the subdirectories to count the number of images in each
#     for subdir in os.listdir(root_folder):
#         subdir_path = os.path.join(root_folder, subdir)
#         if os.path.isdir(subdir_path):
#             subdir_images[subdir] = []
#             for file in os.listdir(subdir_path):
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                     subdir_images[subdir].append(file)
#             min_images = min(min_images, len(subdir_images[subdir]))

#     # Delete extra images in each subdirectory
#     for subdir, images in subdir_images.items():
#         extra_images = len(images) - min_images
#         if extra_images > 0:
#             print(f"Deleting {extra_images} images in {subdir}")
#             images_to_delete = random.sample(images, extra_images)  # Randomly select images for deletion
#             for image in images_to_delete:
#                 os.remove(os.path.join(root_folder, subdir, image))

# # Specify the root folder containing subdirectories with images
# root_folder_path = r"train"

# delete_extra_images(root_folder_path)