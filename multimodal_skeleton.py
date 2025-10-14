import os
import torch
from PIL import Image
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer
from projection import ProjectionLayer
from huggingface_hub import login

torch.backends.cudnn.benchmark = True

hf_token = os.getenv("HF_TOKEN", None)
if hf_token:
    login(token=hf_token)

# Load GPT-OSS
print("Loading GPT-OSS model...")
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
gpt_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

llm_dim = gpt_model.config.hidden_size  # Dynamic: usually 4096 or 5120 for variants
print(f"GPT-OSS loaded (embedding dim: {llm_dim}).")

# Load CLIP Vision
print("Loading CLIP Vision model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
clip_model.to("cuda")
print("CLIP loaded.")

# Load Projection Layer
print("Initializing Projection Layer...")
projection = ProjectionLayer(vision_dim=768, llm_dim=llm_dim).to("cuda")

# Load sample image and test
try:
    image = Image.open("sample_image.jpg")
except FileNotFoundError:
    raise SystemExit("Error: 'sample_image.jpg' not found.")

image_input = preprocess(image).unsqueeze(0).to("cuda")
with torch.no_grad():
    image_features = clip_model.encode_image(image_input)
    print(f"Vision embedding shape: {image_features.shape}")  # [1, 768]
    projected_embeds = projection(image_features)
    print(f"Projected embedding shape: {projected_embeds.shape}")  # [1, llm_dim]

# Tokenize prompt & extract text embeddings
eo_prompt = "Is there visible flooding in the provided satellite image?"
inputs = tokenizer(eo_prompt, return_tensors="pt")
text_embeds = gpt_model.get_input_embeddings()(inputs.input_ids.to(gpt_model.device))  # [1, seq_len, llm_dim]

# Prefix fusion: prepend projected image token embedding
def fuse_embeds_prefix(projected_embeds, text_embeds):
    # projected_embeds: [1, llm_dim], text_embeds: [1, seq_len, llm_dim]
    projected_embeds = projected_embeds.unsqueeze(1)  # [1, 1, llm_dim]
    return torch.cat([projected_embeds, text_embeds], dim=1)

fused_embeds = fuse_embeds_prefix(projected_embeds, text_embeds)
print(f"Fused sequence shape (prefix): {fused_embeds.shape}")  # [1, seq_len+1, llm_dim]

print("SUCCESS! Fusion and projection end-to-end test complete.")

torch.cuda.empty_cache()
