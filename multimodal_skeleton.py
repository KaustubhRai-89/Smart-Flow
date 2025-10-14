# multimodal_skeleton.py - Tests full data flow: Image → Vision → Projection → GPT-Ready
import torch
from PIL import Image
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer
from projection import ProjectionLayer  # Your projection class

# Load GPT-OSS (your working setup)
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
print("GPT-OSS loaded.")

# Load CLIP Vision (your working vision_test.py)
print("Loading CLIP Vision model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
clip_model.to("cuda")
print("CLIP loaded.")

# Load Projection Layer
print("Initializing Projection Layer...")
projection = ProjectionLayer(vision_dim=768, llm_dim=5120).to("cuda")  # Match dims
print("Projection ready.")

# Load and preprocess sample image (your attached one)
try:
    image = Image.open("sample_image.jpg")  # Or rename your attached file
    print("Sample image loaded.")
except FileNotFoundError:
    raise SystemExit("Error: 'sample_image.jpg' not found. Use your attached image.")

image_input = preprocess(image).unsqueeze(0).to("cuda")

# Full Forward Pass Test (No Gradients for Speed)
with torch.no_grad():
    # Step 1: Vision Embedding
    vision_embeds = clip_model.encode_image(image_input)
    print(f"Step 1 - Vision embeds shape: {vision_embeds.shape}")  # [1, 768]

    # Step 2: Project to GPT Space
    projected_embeds = projection(vision_embeds)
    print(f"Step 2 - Projected embeds shape: {projected_embeds.shape}")  # [1, 5120]

    # Step 3: Prep Text Embedding (Simple Demo - Combine with Dummy Text)
    text_prompt = "Based on this satellite image, describe any visible land features."
    text_inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
    text_embeds = gpt_model.model.embeddings(text_inputs.input_ids)  # Get text embeddings
    print(f"Step 3 - Text embeds shape: {text_embeds.shape}")  # e.g., [1, seq_len, 5120]

    # TODO: Full fusion (next phase) - Concat or add projected + text embeds
    print("SUCCESS! Data flows through the entire pipeline without errors.")
    print("Next: Implement text-image fusion and basic generation.")

# Memory Cleanup
del clip_model, gpt_model, projection
torch.cuda.empty_cache()
print("Pipeline test complete. VRAM freed.")
