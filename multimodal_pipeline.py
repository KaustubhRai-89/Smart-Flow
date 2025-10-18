"""
multimodal_pipeline.py - v3.0 FINAL INFERENCE SCRIPT
- Loads the base models (CLIP, Qwen2).
- Loads the trained Projection Layer (Phase 2).
- Loads the trained LoRA Adapters (Phase 3).
- Runs inference on a test image to generate a detailed description.
"""
import os
import torch
import pandas as pd
import io
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import open_clip

# Import PEFT for loading the LoRA adapters
from peft import PeftModel

# Import from your central model definitions
from vision_language_models import ProjectionLayer

# --- Configuration ---
COMPUTE_DTYPE = torch.bfloat16
DEVICE = 'cuda'
PROJECTION_CHECKPOINT = "checkpoints/projection_isro_v1.pt"
LORA_ADAPTER_DIR = "checkpoints/lora_adapter"

class MultimodalPipeline:
    def __init__(self, device=DEVICE):
        self.device = device
        self.vision_model = None
        self.vision_preprocess = None
        self.llm = None
        self.tokenizer = None
        self.projection_layer = None
        self._load_models()

    def _load_models(self):
        print("="*60 + "\nLoading ALL components for FINAL inference...\n" + "="*60)

        # 1. Load Vision Model (Frozen)
        self.vision_model, _, self.vision_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        self.vision_model.to(self.device, dtype=COMPUTE_DTYPE).eval()
        print("✅ 1/4: Vision Encoder loaded.")

        # 2. Load Base Language Model
        llm_id = "Qwen/Qwen2-7B-Instruct"
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
        base_llm = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=quant_config, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ 2/4: Base Language Model loaded.")

        # 3. Load and Apply LoRA Adapters
        print("✅ 3/4: Applying trained LoRA adapters...")
        self.llm = PeftModel.from_pretrained(base_llm, LORA_ADAPTER_DIR)
        print("✅ LoRA adapters merged successfully.")

        # 4. Load Trained Projection Layer
        print("✅ 4/4: Loading trained Projection Layer...")
        self.projection_layer = ProjectionLayer().to(self.device)
        self.projection_layer.load_state_dict(torch.load(PROJECTION_CHECKPOINT, map_location=self.device))
        self.projection_layer.to(dtype=COMPUTE_DTYPE).eval()
        print(f"✅ Projection layer loaded from {PROJECTION_CHECKPOINT}")

        print("\n" + "="*60 + "\n🚀 Fully fine-tuned multimodal pipeline is ready!\n" + "="*60)

    @torch.no_grad()
    def run(self, image: Image, query: str):
        image_tensor = self.vision_preprocess(image).unsqueeze(0).to(self.device).to(COMPUTE_DTYPE)
        vision_embedding = self.vision_model.encode_image(image_tensor)
        projected_embedding = self.projection_layer(vision_embedding)

        messages = [{"role": "system", "content": "You are a satellite imagery analysis expert. Your task is to describe the provided image in detail."}, {"role": "user", "content": query}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        text_embeds = self.llm.get_input_embeddings()(text_inputs.input_ids.to(self.device))

        fused_embeds = torch.cat([projected_embedding.unsqueeze(1), text_embeds], dim=1)
        fused_embeds = fused_embeds.to(self.llm.dtype)
        attention_mask = torch.ones(fused_embeds.shape[:2], dtype=torch.long, device=self.device)

        output_ids = self.llm.generate(inputs_embeds=fused_embeds, attention_mask=attention_mask, max_new_tokens=150, do_sample=False)
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.split("assistant")[-1].strip()

def get_image_from_test_csv(csv_path="data/test.csv", index=100):
    """Utility function to load a single test image from the CSV."""
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    image_bytes_str = row['image']
    image_dict = ast.literal_eval(image_bytes_str)
    image_bytes = image_dict['bytes']
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"Loaded test image: {row['filename']}")
    print(f"One of its true captions is: \"{ast.literal_eval(row['captions'])[0]}\"")
    return image

### Main Execution Block ###
if __name__ == "__main__":
    # Initialize the fully trained pipeline
    pipeline = MultimodalPipeline()

    # Get a test image that the model has never seen before
    test_image = get_image_from_test_csv(index=150) # You can change the index to test different images

    # Ask the model to describe the image
    query = "Analyze this satellite image and provide a detailed description of what you see."
    response = pipeline.run(image=test_image, query=query)

    print("\n" + "="*60)
    print("--- FINAL INFERENCE RESULT ---")
    print("="*60)
    print(f"Query: {query}")
    print(f"Generated Response: {response}")
    print("="*60)
