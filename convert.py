#use the local downladed model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, os, time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model_path = "./gpt-oss-model-local"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

num_total_layers = 24
num_gpu_layers = 12  # safer for 12GB GPU

device_map = {
    "model.embed_tokens": 0,
    **{f"model.layers.{i}": 0 for i in range(num_gpu_layers)},  # GPU
    **{f"model.layers.{i}": "cpu" for i in range(num_gpu_layers, num_total_layers)},  # CPU
    "model.norm": "cpu",
    "lm_head": "cpu"
}

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device_map,
    attn_implementation="eager"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [{"role": "user", "content": "Explain what MXFP4 quantization is in simple terms."}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)  # keep on CPU

torch.set_grad_enabled(False)
start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=2,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
end = time.time()

generated = outputs[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
print(f"\nElapsed time: {end - start:.2f}s")
