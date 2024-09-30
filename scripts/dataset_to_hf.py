# Uploads a locally recorded LuckyRobot dataset to HuggingFace
import os
import json
from datasets import Dataset, DatasetDict
from PIL import Image
from pathlib import Path

# Define the data directory
data_dir = Path("/home/oop/dev/luckyrobots/Binary/092924/luckyrobots/robotdata")

# Initialize the structured data container
data_entries = []

# Gather all files in the directory
files = sorted(data_dir.glob("*"))

# Helper function to read image files
def load_image(file_path):
    with open(file_path, "rb") as f:
        return f.read()

# Helper function to read position text files
def load_position(file_path):
    with open(file_path, "r") as f:
        return json.loads(f.read())

# Group files by the unique prefix
prefix_groups = {}
for file in files:
    if file.is_file():
        prefix = file.stem.split("_")[0]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = {}
        if "depth" in file.stem:
            prefix_groups[prefix][f"{file.stem.split('_')[1]}_depth_image"] = load_image(file)
        elif "rgb" in file.stem:
            prefix_groups[prefix][f"{file.stem.split('_')[1]}_rgb_image"] = load_image(file)
        elif "pos" in file.stem:
            prefix_groups[prefix][f"{file.stem.split('_')[1]}_pos"] = load_position(file)

# Create dataset entries
for prefix, data in prefix_groups.items():
    entry = {
        "id": prefix,
        "cam1_depth_image": data.get("cam1_depth_image"),
        "cam1_rgb_image": data.get("cam1_rgb_image"),
        "cam2_depth_image": data.get("cam2_depth_image"),
        "cam2_rgb_image": data.get("cam2_rgb_image"),
        "hand_pos": data.get("Hand_pos", {}),
        "body_pos": data.get("Body_pos", {}),
        "head_pos": data.get("Head_pos", {})
    }
    data_entries.append(entry)

# Create a Hugging Face Dataset object
dataset = Dataset.from_dict({"entries": data_entries})

# Save to a JSON file for upload
output_path = data_dir / "robot_simulation_dataset.json"
dataset.to_json(output_path)

print(f"Dataset saved to: {output_path}")
