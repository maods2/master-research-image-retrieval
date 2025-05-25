import os
import shutil
from pathlib import Path
from random import sample
from tqdm import tqdm
import pandas as pd

def split_dataset_multi_folder(source_dir, target_dir, sample_size_per_class=10):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"

    # Create target directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through each label (subfolder)
    for label_dir in source_dir.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
            images = []
            for ext in image_extensions:
                images.extend(label_dir.rglob(ext))

            # Ensure there are enough images to sample
            if len(images) < sample_size_per_class:
                print(f"Not enough images in {label} to sample {sample_size_per_class}. Skipping...")
                continue

            # Sample test images
            test_images = sample(images, sample_size_per_class)
            train_images = [img for img in images if img not in test_images]

            # Create label subdirectories in train and test folders
            (train_dir / label).mkdir(parents=True, exist_ok=True)
            (test_dir / label).mkdir(parents=True, exist_ok=True)

            # Copy test images
            for img_path in tqdm(test_images, desc=f"Copying test images for {label}"):
                shutil.copy(img_path, test_dir / label / img_path.name)

            # Copy train images
            for img_path in tqdm(train_images, desc=f"Copying train images for {label}"):
                shutil.copy(img_path, train_dir / label / img_path.name)

    print("Dataset split completed.")

def split_dataset_single_folder(source_dir, target_dir, csv_file, sample_size_per_class=10):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"

    # Create target directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv(csv_file)
    image_to_label = {row['image_id']: row['dx'] for _, row in data.iterrows()}

    # Scan the folder for images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF']
    images = []
    for ext in image_extensions:
        images.extend(source_dir.rglob(ext))

    # Map images to their labels
    label_to_images = {}
    for img_path in images:
        image_id = img_path.stem
        if image_id in image_to_label:
            label = image_to_label[image_id]
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img_path)

    # Process each label
    for label, images in label_to_images.items():
        # Ensure there are enough images to sample
        if len(images) < sample_size_per_class:
            print(f"Not enough images in {label} to sample {sample_size_per_class}. Skipping...")
            continue

        # Sample test images
        test_images = sample(images, sample_size_per_class)
        train_images = [img for img in images if img not in test_images]

        # Create label subdirectories in train and test folders
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (test_dir / label).mkdir(parents=True, exist_ok=True)

        # Copy test images
        for img_path in tqdm(test_images, desc=f"Copying test images for {label}"):
            try:
                shutil.copy(img_path, test_dir / label / img_path.name)
            except FileNotFoundError:
                print(f"File not found: {img_path}. Skipping...")

        # Copy train images
        for img_path in tqdm(train_images, desc=f"Copying train images for {label}"):
            try:
                shutil.copy(img_path, train_dir / label / img_path.name)
            except FileNotFoundError:
                print(f"File not found: {img_path}. Skipping...")

    print("Dataset split completed.")

if __name__ == "__main__":
    import sys
    import os
    
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        print("No arguments provided. Please provide the source directory and target directory.")
        sys.exit(1)
    if len(arguments) > 1:
        print("Too many arguments provided. Please provide only the source directory and target directory.")
        sys.exit(1)

    if arguments[0] == "ovarian-cancer":
        split_dataset_multi_folder(
            source_dir="./datasets/raw/OvarianCancer",
            target_dir="./datasets/final/ovarian-cancer-splitted",
            sample_size_per_class=20
        )
    elif arguments[0] == "skin-cancer":
        split_dataset_single_folder(
            source_dir="./datasets/raw/Skin-Cancer",
            target_dir="./datasets/final/skin-cancer-splitted",
            csv_file="./datasets/raw/Skin-Cancer/HAM10000_metadata.csv",
            sample_size_per_class=20
        )
    else:
        print("Invalid argument. Please provide 'ovarian-cancer' or 'skin-cancer'.")
        sys.exit(1)
