import os
import json
from datasets import Dataset
from PIL import Image

# Directory containing your dataset files
DATA_DIR = "/home/oop/dev/luckyrobots/Binary/092924/luckyrobots/robotdata"
DATASET_NAME = "hu-po/lr-test"

# Initialize the structured data container
data_entries = []

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
for file in os.scandir(DATA_DIR):
    if file.is_file():
        prefix = os.path.splitext(file.name)[0].split("_")[0]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = {}
        if "depth" in file.name:
            prefix_groups[prefix][f"{file.name.split('_')[1]}_depth_image"] = load_image(file.path)
        elif "rgb" in file.name:
            prefix_groups[prefix][f"{file.name.split('_')[1]}_rgb_image"] = load_image(file.path)
        elif "pos" in file.name:
            prefix_groups[prefix][f"{file.name.split('_')[1]}_pos"] = load_position(file.path)

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
output_path = os.path.join(DATA_DIR, "robot_simulation_dataset.json")
dataset.to_json(output_path)
print(f"Dataset saved to: {output_path}")

print("Pushing dataset to Hugging Face Hub...")
dataset.push_to_hub(DATASET_NAME)
print("Dataset pushed to Hugging Face Hub successfully.")