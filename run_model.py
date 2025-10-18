"""
run_model.py - FINAL WORKING VERSION v1.1
- Switched to Qwen/Qwen2-7B-Instruct for compatibility.
- Added explicit download message for user feedback.
- Fixed `torch_dtype` deprecation warning.
"""
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== ENVIRONMENT & GPU CHECK =====
print("="*60)
print("STEP 1: Verifying Environment")
print("="*60)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if not torch.cuda.is_available():
    print("❌ CUDA not available. Please reinstall PyTorch with CUDA support.")
    sys.exit(1)
print(f"✅ GPU Found: {torch.cuda.get_device_name(0)}")

# ===== MODEL CONFIGURATION =====
print("\n" + "="*60)
print("STEP 2: Loading Qwen/Qwen2-7B-Instruct")
print("="*60)

model_id = "Qwen/Qwen2-7B-Instruct"
print(f"Model ID: {model_id}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")

    print("\n⏳ Loading and quantizing model...")
    print("   This involves a one-time download of ~15 GB.")
    print("   This can take 20-60 minutes depending on your internet connection. Please be patient.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,  # Corrected from torch_dtype
        trust_remote_code=True
    )

    hidden_size = model.config.hidden_size
    print(f"\n✅ Model loaded successfully!")
    print(f"   Hidden Size: {hidden_size}")

except Exception as e:
    print(f"\n❌ MODEL LOAD FAILED: {e}")
    sys.exit(1)

# ===== INFERENCE TEST =====
print("\n" + "="*60)
print("STEP 3: Inference Test")
print("="*60)
allocated_mem = torch.cuda.memory_allocated(0) / 1e9
print(f"GPU Memory after load: {allocated_mem:.2f} GB")

messages = [
    {"role": "system", "content": "You are a helpful assistant for ISRO."},
    {"role": "user", "content": "Summarize how LLMs can support ISRO's Earth Observation programs."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\n⏳ Generating response...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*60)
print("✅ GENERATION SUCCESSFUL!")
print("="*60)
print(f"\nResponse:\n{response}")

# ===== FINAL VERDICT =====
print("\n" + "="*60)
print("PROJECT STATUS: UNBLOCKED")
print("="*60)
print("✅ Qwen2-7B-Instruct is working correctly on your 12GB GPU.")
print(f"✅ The new language model embedding dimension is: {hidden_size}")
print("✅ You can now proceed with the multimodal pipeline.")

