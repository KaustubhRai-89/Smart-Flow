# run_model.py (Updated for GPT-OSS Pre-Quantized Loading)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Verification Step ---
if not torch.cuda.is_available():
    raise SystemExit("Error: PyTorch cannot find your GPU. Please check your NVIDIA Driver, CUDA, and cuDNN installations.")
print(f"Success! PyTorch is using your GPU: {torch.cuda.get_device_name(0)}")
# -------------------------

# The official model ID on Hugging Face
model_name = "EleutherAI/gpt-neox-20b"

print("Loading tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Configuration for your 12GB GPU ---
print("Loading GPT-OSS-20B model. This will take time and disk space on the first run...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # Changed from torch_dtype to dtype
    device_map="auto",           # Automatically place on GPU
    trust_remote_code=True,      # For any custom model code
    low_cpu_mem_usage=True,      # Saves CPU memory during loading
)
# ---------------------------------------------

print("Model has been loaded successfully.")

# A test prompt related to our project
prompt = "In the context of multimodal AI, what is a projection layer used for?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate a response
print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)  # Added sampling for better output
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Model Response ---")
print(response_text)
print("----------------------")
