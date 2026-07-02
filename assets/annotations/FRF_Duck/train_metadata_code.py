import pandas
import random
from pathlib import Path
from PIL import Image


train_path = Path("assets/annotations/FRF_Duck/train/images")
images_data = []

for path_item in train_path.iterdir():
    # Omit directories
    if path_item.is_dir():
        print(f"Skipping directory: {path_item}")
        continue

    # Image size with no loading it
    with Image.open(path_item) as img:
        size = img.size
    
    # Get the serie from the filename
    # Expected format: FRF_<camera>_snap_<serie>_EBG.jpg
    serie = path_item.stem.split("_")[3]

    # Append the metadata
    images_data.append({
        "File": path_item.name, 
        "Size": size, 
        "Height": size[0], 
        "Width": size[1], 
        "Serie": serie 
    })

# Save training metadata to CSV
df = pandas.DataFrame(images_data)
df.to_csv("assets/annotations/FRF_Duck/train_metadata.csv", index=False)

# -------------------------------------------------------------------------

# Select 20 random training images
image_files = list(train_path.glob("*.jpg"))
image_files = random.sample(population=image_files, k=20)

# Save a txt file with the selected training images
output_file = Path("assets/annotations/FRF_Duck/train_image_files.txt")
with open(output_file, "w") as f:
    for image_file in image_files:
        f.write(f"{image_file.name}\n")