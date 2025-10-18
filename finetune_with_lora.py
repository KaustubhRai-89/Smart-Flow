"""
finetune_with_lora.py - v3.2 FINAL & WORKING
- Corrects the final dtype mismatch error during the LoRA training loop.
- This script is now ready for the final training run.
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
import io
import ast

# Import PEFT library for LoRA
from peft import get_peft_model, LoraConfig, TaskType

# Import from your central model definitions
from vision_language_models import ProjectionLayer

# --- Configuration ---
COMPUTE_DTYPE = torch.bfloat16
DEVICE = 'cuda'

# --- Training Hyperparameters ---
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
PROJECTION_CHECKPOINT = "checkpoints/projection_isro_v1.pt"
LORA_CHECKPOINT_DIR = "checkpoints/lora_adapter"

# --- Re-define the Dataset class locally for this script ---
class ISRO_Dataset_From_CSV(Dataset):
    def __init__(self, csv_file, preprocess, tokenizer):
        try:
            self.data = pd.read_csv(csv_file).explode('captions').reset_index(drop=True)
        except FileNotFoundError:
            print(f"❌ ERROR: The dataset CSV file was not found at '{csv_file}'")
            raise
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        print(f"✅ Dataset loaded and processed. Total pairs: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_dict = ast.literal_eval(row['image'])
        image_bytes = image_dict['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = self.preprocess(image)
        caption = row['captions']
        tokenized_caption = self.tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")["input_ids"].squeeze()
        return processed_image, tokenized_caption

# --- Main Fine-Tuning Function ---
def finetune():
    print("=" * 60 + "\nStarting Phase 3: End-to-End Fine-Tuning with LoRA\n" + "=" * 60)

    # 1. Load All Models
    vision_model, _, vision_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    vision_model.to(DEVICE, dtype=COMPUTE_DTYPE).eval()
    for param in vision_model.parameters():
        param.requires_grad = False
    print("✅ Vision model loaded and frozen.")

    llm_id = "Qwen/Qwen2-7B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    llm = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=quant_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Language model loaded.")

    projection_layer = ProjectionLayer().to(DEVICE, dtype=COMPUTE_DTYPE)
    projection_layer.load_state_dict(torch.load(PROJECTION_CHECKPOINT, map_location=DEVICE))
    projection_layer.train()
    print(f"✅ Projection layer loaded from {PROJECTION_CHECKPOINT} and unfrozen.")

    # 2. Configure and Apply LoRA
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    llm = get_peft_model(llm, lora_config)
    print("\nLoRA adapters applied to the language model:")
    llm.print_trainable_parameters()

    # 3. Prepare Dataset
    csv_file_path = "data/train.csv"
    dataset = ISRO_Dataset_From_CSV(csv_file_path, vision_preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. Setup Optimizer
    trainable_params = list(projection_layer.parameters()) + list(llm.parameters())
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, trainable_params), lr=LEARNING_RATE)

    # 5. Fine-Tuning Loop
    print("\n" + "=" * 60 + "\nStarting Fine-Tuning...\n" + "=" * 60)
    llm.train()
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, captions in progress_bar:
            images = images.to(DEVICE, dtype=COMPUTE_DTYPE)
            captions = captions.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                image_embeddings = vision_model.encode_image(images)
            projected_embeddings = projection_layer(image_embeddings)

            caption_embeds = llm.get_input_embeddings()(captions)
            inputs_embeds = torch.cat([projected_embeddings.unsqueeze(1), caption_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE)
            labels = torch.cat([torch.full((images.size(0), 1), -100, dtype=torch.long, device=DEVICE), captions], dim=1)

            # CRITICAL FIX: Ensure the final input tensor matches the LLM's expected dtype
            inputs_embeds = inputs_embeds.to(llm.dtype)

            # Forward pass
            outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"LM Loss": f"{loss.item():.4f}"})

    # 6. Save the LoRA Adapters
    print("\n✅ Fine-tuning complete. Saving LoRA adapters...")
    llm.save_pretrained(LORA_CHECKPOINT_DIR)
    tokenizer.save_pretrained(LORA_CHECKPOINT_DIR)
    print(f"✅ LoRA adapters saved to: {LORA_CHECKPOINT_DIR}")

if __name__ == "__main__":
    finetune()
