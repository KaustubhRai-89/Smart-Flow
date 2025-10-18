import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

model_id = "openai/gpt-oss-20b"
save_path = './gpt-oss-model-local'
try:

    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)


    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


except Exception as e:
    print(e)
