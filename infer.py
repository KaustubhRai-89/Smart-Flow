import torch
import os
from PIL import Image
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from peft import PeftModel
from typing import Dict

# ===============================
# 1. CONFIGURATION
# ===============================

LLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
VISION_TOWER = "openai/clip-vit-large-patch14"
LLM_EMBED_SIZE = 1536  # Must match DeepSeek hidden size
VISION_HIDDEN_SIZE = 1024  # CLIP ViT-L/14 hidden size
IMG_TOKEN = "<image>"

# Paths
LORA_ADAPTER_PATH = "./llava_deepseek_lora_adapters"
PROJECTOR_PATH = "./llava_deepseek_projector.pt"

# Inference Test Setup
TEST_IMAGE_PATH = "test_image.jpg"
QUESTION = "What does this satellite image show? ASSISTANT:"
GENERATION_KWARGS = {
    "max_new_tokens": 120,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
}

# ===============================
# 2. MODEL DEFINITION
# ===============================

class LLaVADeepSeek(nn.Module):
    def __init__(self, language_model, vision_tower, mm_projector):
        super().__init__()
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector

    @torch.no_grad()
    def generate_answer(self, image_tensor: torch.Tensor, tokenized_prompt: Dict[str, torch.Tensor], tokenizer: AutoTokenizer) -> str:
        self.eval()
        device = next(self.language_model.parameters()).device

        input_ids = tokenized_prompt["input_ids"].to(device)
        image_token_id = tokenizer.convert_tokens_to_ids(IMG_TOKEN)

        # Use the **base model** embeddings (important for PEFT)
        base_model = self.language_model.get_base_model()
        base_embed_layer = base_model.get_input_embeddings()
        input_embeds = base_embed_layer(input_ids)

        # Encode image
        image_features = self.vision_tower(
            image_tensor.to(device, dtype=torch.bfloat16)
        ).last_hidden_state

        # Project to LLM dimension
        image_embeddings = self.mm_projector(image_features)

        # Find where to insert the <image> embeddings
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
        if len(image_token_indices[1]) == 0:
            raise ValueError("No <image> token found in the prompt!")

        img_pos = image_token_indices[1][0].item()

        # Replace the single image token embedding with all patch embeddings
        pre = input_embeds[:, :img_pos]
        post = input_embeds[:, img_pos + 1:]
        spliced_embeds = torch.cat([pre, image_embeddings, post], dim=1)

        # Create attention mask for new sequence length
        attention_mask = torch.ones(
            (spliced_embeds.shape[0], spliced_embeds.shape[1]),
            dtype=torch.long,
            device=device
        )

        # Generate
        outputs = self.language_model.generate(
            inputs_embeds=spliced_embeds,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **GENERATION_KWARGS
        )

        # Decode only newly generated part
        prompt_len = spliced_embeds.shape[1]
        new_tokens = outputs[0, prompt_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return answer


# ===============================
# 3. MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    torch.cuda.empty_cache()

    print("🔹 Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': [IMG_TOKEN]})

    image_processor = CLIPImageProcessor.from_pretrained(VISION_TOWER)

    print(f"🔹 Loading base LLM ({LLM_MODEL_NAME})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    base_llm.resize_token_embeddings(len(tokenizer))

    print("🔹 Loading vision tower...")
    vision_tower = CLIPVisionModel.from_pretrained(
        VISION_TOWER,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("🔹 Loading LoRA adapters...")
    language_model = PeftModel.from_pretrained(base_llm, LORA_ADAPTER_PATH)
    language_model.eval()

    print("🔹 Loading projector...")
    mm_projector = nn.Linear(VISION_HIDDEN_SIZE, LLM_EMBED_SIZE, dtype=torch.bfloat16)
    state_dict = torch.load(PROJECTOR_PATH, map_location="cpu")
    mm_projector.load_state_dict(state_dict)
    mm_projector = mm_projector.to(language_model.device, dtype=torch.bfloat16)
    mm_projector.eval()

    # Build the multimodal model
    model = LLaVADeepSeek(language_model, vision_tower, mm_projector)

    for p in vision_tower.parameters():
        p.requires_grad = False

    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"❌ Please provide a valid image at '{TEST_IMAGE_PATH}'")

    print(f"\n🖼️  Loading image: {TEST_IMAGE_PATH}")
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"]

    prompt = f"{IMG_TOKEN}USER: {QUESTION}"
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")

    print("\n🚀 Generating answer...")
    answer = model.generate_answer(image_tensor, tokenized_prompt, tokenizer)

    print("\n" + "=" * 50)
    print(f"❓ Question: {QUESTION}")
    print(f"🤖 Answer: {answer}")
    print("=" * 50)
