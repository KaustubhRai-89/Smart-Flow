from datasets import load_dataset
import os

base_path = "LLaVA_Instruct_150K"

# 1. Load a single JSON file (e.g., complex_reasoning_77k.json)
# Use the 'json' builder and provide the relative path to the file.
dataset_single = load_dataset(
    'json', 
    data_files=os.path.join(base_path, 'complex_reasoning_77k.json'),
    split='train'
)

print("--- Loaded Single File (77K) ---")
print(f"Total examples: {len(dataset_single)}")
print(dataset_single[0])

# ----------------------------------------------------------------------
# 2. Load ALL relevant LLaVA annotation files together
# The LLaVA-Instruct-150K dataset is a combination of these three files.
data_files_all = {
    'train': [
        os.path.join(base_path, 'conversation_58k.json'),
        os.path.join(base_path, 'detail_23k.json'),
        os.path.join(base_path, 'complex_reasoning_77k.json')
    ]
}

dataset_full = load_dataset(
    'json', 
    data_files=data_files_all, 
    split='train'
)

print("\n--- Loaded All Files (LLaVA-Instruct-150K) ---")
print(f"Total examples: {len(dataset_full)}")
print(dataset_full[0])


# For GPU check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")