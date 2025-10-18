"""
train_projection.py - v2.1 FINAL - REAL DATA TRAINING
This script trains the projection layer on the real RSICD dataset.
- It includes a custom Dataset class that reads image bytes directly from the Kaggle CSV.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import open_clip
from tqdm import tqdm
import pandas as pd
import io  # Required to read bytes from memory
import ast # Required to safely evaluate the string representation of the image bytes

# Import the ProjectionLayer from your central model definitions file
from vision_language_models import ProjectionLayer

# --- Configuration ---
COMPUTE_DTYPE = torch.bfloat16
DEVICE = 'cuda'

# --- Training Hyperparameters ---
EPOCHS = 10 # Start with 10 epochs for this large dataset
BATCH_SIZE = 8 # Adjust based on your VRAM
LEARNING_RATE = 1e-4
CHECKPOINT_NAME = "projection_isro_v1.pt"

# --- Real ISRO Dataset Class (Reads Bytes from CSV) ---
class ISRO_Dataset_From_CSV(Dataset):
    """
    A custom PyTorch Dataset that loads image bytes directly from a CSV column.
    """
    def __init__(self, csv_file, preprocess, tokenizer):
        try:
            self.data = pd.read_csv(csv_file)
            # The dataset has multiple captions per image. We need to "unroll" it.
            # This creates a new dataframe where each row is a unique image-caption pair.
            self.data = self.data.explode('captions').reset_index(drop=True)
        except FileNotFoundError:
            print(f"❌ ERROR: The dataset CSV file was not found at '{csv_file}'")
            raise

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        print(f"✅ Dataset loaded and processed. Total image-caption pairs: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- Load image directly from the byte string in the 'image' column ---
        image_bytes_str = row['image']
        # Safely evaluate the string "{'bytes': b'...'}" to a dictionary
        image_dict = ast.literal_eval(image_bytes_str)
        # Get the actual bytes
        image_bytes = image_dict['bytes']
        # Open the bytes as if it were a file in memory
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = self.preprocess(image)

        # --- Load the corresponding caption ---
        caption = row['captions']
        tokenized_caption = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )["input_ids"].squeeze()

        return processed_image, tokenized_caption

# --- Loss Function (remains the same) ---
def normalized_cosine_loss(image_embeddings, text_embeddings):
    image_norm = F.normalize(image_embeddings, p=2, dim=-1)
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    loss = 1 - (image_norm * text_norm).sum(dim=-1).mean()
    return loss

# --- Main Training Function ---
def train():
    print("="*60 + "\nStarting Phase 2: Projection Layer Training on Real RSICD Dataset\n" + "="*60)

    # 1. Load Models
    vision_model, _, vision_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    vision_model.to(DEVICE, dtype=COMPUTE_DTYPE).eval()

    llm_id = "Qwen/Qwen2-7B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    llm = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)

    projection_layer = ProjectionLayer().to(DEVICE, dtype=COMPUTE_DTYPE)

    # 2. Freeze Large Models
    for param in vision_model.parameters():
        param.requires_grad = False
    for param in llm.parameters():
        param.requires_grad = False

    projection_layer.train()
    print("✅ CLIP and Qwen2 models are frozen. Only the projection layer is trainable.")

    # 3. Prepare Real Dataset and DataLoader
    # IMPORTANT: We use train.csv for training.
    csv_file_path = "data/train.csv"
    dataset = ISRO_Dataset_From_CSV(csv_file_path, vision_preprocess, tokenizer)
    # Set num_workers=0 on Windows to avoid multiprocessing issues
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(projection_layer.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("\n" + "="*60 + "\nStarting Training on RSICD Dataset...\n" + "="*60)
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, captions in progress_bar:
            images = images.to(DEVICE, dtype=COMPUTE_DTYPE)
            captions = captions.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                image_embeddings = vision_model.encode_image(images)
                text_embeddings = llm.get_input_embeddings()(captions).mean(dim=1)
            projected_embeddings = projection_layer(image_embeddings)
            loss = normalized_cosine_loss(projected_embeddings, text_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # 6. Save the Final Model Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{CHECKPOINT_NAME}"
    torch.save(projection_layer.state_dict(), checkpoint_path)
    print("\n" + "="*60)
    print(f"✅ Training complete. Trained model saved to: {checkpoint_path}")
    print("="*60)

if __name__ == "__main__":
    train()

