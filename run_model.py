import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Enable cuDNN autotune for speed on uniform inputs
torch.backends.cudnn.benchmark = True

# Hugging Face authentication (set HF_TOKEN env if gated)
hf_token = os.getenv("HF_TOKEN", None)
if hf_token:
    login(token=hf_token)

# --- GPU & Model Verification ---
if not torch.cuda.is_available():
    raise SystemExit("Error: PyTorch cannot find your GPU. Please check NVIDIA Driver, CUDA, cuDNN.")

print(f"Success! PyTorch is using your GPU: {torch.cuda.get_device_name(0)}")

model_name = "openai/gpt-oss-20b"
print("Loading tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

try:
    print("Loading GPT-OSS-20B model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"Error loading GPT-OSS model: {str(e)}")
    exit(1)

print(f"Model loaded! Hidden size: {model.config.hidden_size}")

prompt = "Summarize the purpose of integrating vision models and LLMs for ISRO Earth Observation."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=60, do_sample=True)
    print("Generated output:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

torch.cuda.empty_cache()
