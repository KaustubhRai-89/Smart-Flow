"""
train_projection.py - Phase 2: Alignment Training

This script trains ONLY the projection layer to align the vision and language spaces.
- It freezes the large vision (CLIP) and language (Qwen2) models.
- It uses a Cosine Similarity Loss to teach the projection layer to map an
  image embedding to be as close as possible to its corresponding text caption embedding.
- This is a memory-efficient training stage designed for limited hardware.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import open_clip
from tqdm import tqdm  # For a nice progress bar

# Import the custom projection layer from your pipeline
from multimodal_pipeline import ProjectionLayer

# --- Configuration ---
COMPUTE_DTYPE = torch.bfloat16
DEVICE = 'cuda'

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 4  # Adjust based on your VRAM
LEARNING_RATE = 1e-4


# --- Dummy Dataset for Initial Testing ---
# In a real scenario, this would load your ISRO image-caption pairs
class DummyDataset(Dataset):
    def __init__(self, preprocess, tokenizer, num_samples=100):
        self.num_samples = num_samples
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        # Create a dummy image and some dummy captions
        self.dummy_image = Image.new('RGB', (224, 224), color='blue')
        self.captions = [
            "a blue square",
            "an image of the color blue",
            "a photo of a blue canvas",
            "deep blue sky"
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.preprocess(self.dummy_image)
        caption = self.captions[idx % len(self.captions)]
        tokenized_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=32,  # Keep text length fixed
            return_tensors="pt"
        )["input_ids"].squeeze()
        return image, tokenized_caption


# --- Loss Function ---
def normalized_cosine_loss(image_embeddings, text_embeddings):
    image_norm = F.normalize(image_embeddings, p=2, dim=-1)
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    loss = 1 - (image_norm * text_norm).sum(dim=-1).mean()
    return loss


# --- Main Training Function ---
def train():
    print("=" * 60 + "\nStarting Phase 2: Projection Layer Training\n" + "=" * 60)

    # 1. Load All Models
    # Vision Model (CLIP)
    vision_model, _, vision_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='laion2b_s32b_b82k'
    )
    vision_model.to(DEVICE, dtype=COMPUTE_DTYPE).eval()

    # Language Model (Qwen2)
    llm_id = "Qwen/Qwen2-7B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_id, quantization_config=quant_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_id)

    # Projection Layer (This is the only model we will train)
    projection_layer = ProjectionLayer().to(DEVICE, dtype=COMPUTE_DTYPE)

    # 2. Freeze Large Models
    for param in vision_model.parameters():
        param.requires_grad = False
    for param in llm.parameters():
        param.requires_grad = False

    # Ensure only the projection layer is trainable
    projection_layer.train()
    print("✅ CLIP and Qwen2 models are frozen. Only the projection layer is trainable.")

    # 3. Prepare Dataset and DataLoader
    dataset = DummyDataset(vision_preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Dummy dataset created with {len(dataset)} samples.")

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(projection_layer.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("\n" + "=" * 60 + "\nStarting Training...\n" + "=" * 60)
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, captions in progress_bar:
            images = images.to(DEVICE, dtype=COMPUTE_DTYPE)
            captions = captions.to(DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():
                image_embeddings = vision_model.encode_image(images)
                # Get embeddings for text, using mean pooling
                text_embeddings = llm.get_input_embeddings()(captions).mean(dim=1)

            # Project vision embeddings into the LLM space
            projected_embeddings = projection_layer(image_embeddings)

            # Calculate loss
            loss = normalized_cosine_loss(projected_embeddings, text_embeddings)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # 6. Save the Trained Model
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/projection_v1.pt"
    torch.save(projection_layer.state_dict(), checkpoint_path)
    print("\n" + "=" * 60)
    print(f"✅ Training complete. Trained projection layer saved to: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
