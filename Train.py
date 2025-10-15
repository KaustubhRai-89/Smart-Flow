import torch
import json
import os
import re
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Sequence, Any, Union

# Hugging Face Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    CLIPVisionModel,
    CLIPImageProcessor
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from torch.nn import Linear

# --- 1. Configuration and Paths ---
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
IMAGE_BASE_PATH = "" 
LLAVA_JSON_PATH = "satellite.json"

# Multimodal Configuration
VISION_TOWER = "openai/clip-vit-large-patch14"
# CORRECTED a dimension mismatch. The model's hidden size is 1536, not 2048.
LLM_EMBED_SIZE = 1536
VISION_HIDDEN_SIZE = 1024
IMG_TOKEN = "<image>"

# PEFT/QLoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- 2. Custom Multimodal Model (LLaVA-Style) ---

class LLaVADeepSeek(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        
        # 1. Load Base LLM
        self.language_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            # FIXED deprecation warning: torch_dtype -> dtype
            dtype=torch.bfloat16
        )
        self.config = self.language_model.config
        
        # 2. Load and Freeze Vision Encoder
        self.vision_tower = CLIPVisionModel.from_pretrained(VISION_TOWER)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        
        # 3. Projection Layer (MLP Connector)
        self.mm_projector = Linear(VISION_HIDDEN_SIZE, LLM_EMBED_SIZE, device=self.language_model.device)

        # 4. Apply LoRA to the internal LLM
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=TARGET_MODULES,
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.config.use_cache = False
        
    def gradient_checkpointing_enable(self, **kwargs):
        """Passes the checkpointing command to the inner language model."""
        if hasattr(self.language_model, "gradient_checkpointing_enable"):
            self.language_model.gradient_checkpointing_enable(**kwargs)
            
    def forward(self, input_ids: torch.Tensor, image_tensors: torch.Tensor, labels: torch.Tensor = None, **kwargs) -> Any:
        # 1. Get the language model's input embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 2. Encode Image and Project Features
        with torch.no_grad():
            image_features = self.vision_tower(image_tensors.to(self.vision_tower.device)).last_hidden_state
        
        image_embeddings = self.mm_projector(image_features)

        # 3. Combine Text and Image Embeddings
        new_input_embeds = []
        new_labels = []
        
        image_token_id = tokenizer.convert_tokens_to_ids(IMG_TOKEN)

        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == image_token_id)[0]
            
            if len(image_token_indices) == 0:
                new_input_embeds.append(input_embeds[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue
                
            image_token_start_pos = image_token_indices[0]
            
            cur_input_embeds_pre = input_embeds[batch_idx, :image_token_start_pos]
            cur_input_embeds_post = input_embeds[batch_idx, image_token_start_pos + 1:]
            
            spliced_embeds = torch.cat(
                [cur_input_embeds_pre, image_embeddings[batch_idx], cur_input_embeds_post], 
                dim=0
            )
            new_input_embeds.append(spliced_embeds)

            # 4. Adjust Labels for the new sequence length
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_labels_pre = cur_labels[:image_token_start_pos]
                
                image_labels = torch.full(
                    (image_embeddings.shape[1],), -100, 
                    device=labels.device, 
                    dtype=labels.dtype
                )
                
                cur_labels_post = cur_labels[image_token_start_pos + 1:]
                
                spliced_labels = torch.cat(
                    [cur_labels_pre, image_labels, cur_labels_post], 
                    dim=0
                )
                new_labels.append(spliced_labels)

        # 5. Pad the combined sequences and pass to the LLM
        max_len = max(x.shape[0] for x in new_input_embeds)
        
        padded_embeds = torch.zeros(
            (len(new_input_embeds), max_len, new_input_embeds[0].shape[1]),
            dtype=new_input_embeds[0].dtype,
            device=new_input_embeds[0].device
        )
        attention_mask = torch.zeros(len(new_input_embeds), max_len, device=padded_embeds.device, dtype=torch.long)

        for i, x in enumerate(new_input_embeds):
            padded_embeds[i, :x.shape[0], :] = x
            attention_mask[i, :x.shape[0]] = 1
            
        padded_labels = None
        if labels is not None and len(new_labels) > 0:
            padded_labels = torch.full(
                (len(new_labels), max_len), -100, 
                dtype=new_labels[0].dtype, 
                device=new_labels[0].device
            )
            for i, x in enumerate(new_labels):
                padded_labels[i, :x.shape[0]] = x

        output = self.language_model(
            inputs_embeds=padded_embeds,
            attention_mask=attention_mask,
            labels=padded_labels
        )
        
        return output

    def to(self, *args, **kwargs):
        self.language_model.to(*args, **kwargs)
        self.vision_tower.to(*args, **kwargs)
        self.mm_projector.to(*args, **kwargs)
        return self

# --- 3. Custom Data Collator and Preprocessor ---

@dataclass
class MultimodalDataCollator:
    tokenizer: AutoTokenizer
    image_processor: CLIPImageProcessor
    
    def unwrap_tensor(self, item: Union[torch.Tensor, list]) -> torch.Tensor:
        while isinstance(item, list) and len(item) == 1:
            item = item[0]
        if not isinstance(item, torch.Tensor):
            try:
                return torch.tensor(item)
            except:
                raise TypeError(f"Could not convert item to Tensor in collator. Item type: {type(item)}")
        return item
        
    def __call__(self, examples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(e['input_ids']) for e in examples]
        labels = [torch.tensor(e['labels']) for e in examples]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) 
        
        image_tensors = torch.stack([self.unwrap_tensor(e['image_tensors']) for e in examples])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'image_tensors': image_tensors
        }

def process_data_example(example, tokenizer, image_processor):
    if 'filepath' not in example or not example.get('captions'):
        return None 
    
    try:
        image_path = os.path.join(IMAGE_BASE_PATH, example['filepath'])
        if not os.path.exists(image_path):
            return None

        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = image_processor(images=raw_image, return_tensors='pt')['pixel_values'].squeeze(0) 
        
        caption = example['captions'][0].strip()
        caption = re.sub(r"\[\'?|\'\]|\]|\[", "", caption).strip()
        
        prompt = f"{IMG_TOKEN}USER: What does this satellite image show? ASSISTANT: {caption}{tokenizer.eos_token}"
        
        tokenized = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH)
        
        return {
            "input_ids": tokenized["input_ids"], 
            "labels": tokenized["input_ids"], 
            "image_tensors": image_tensor
        }
        
    except Exception as e:
        print(f"Error processing example {example.get('filepath', 'N/A')}: {e}. Skipping.")
        return None

# --- 4. Main Execution ---

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.add_special_tokens({'additional_special_tokens': [IMG_TOKEN]})

    image_processor = CLIPImageProcessor.from_pretrained(VISION_TOWER)
    custom_model = LLaVADeepSeek(None, tokenizer)

    custom_model.language_model.resize_token_embeddings(len(tokenizer))

    with open(LLAVA_JSON_PATH, 'r') as f:
        data = json.load(f)

    hf_dataset = Dataset.from_list(data)
    
    processed_dataset = hf_dataset.map(
        lambda example: process_data_example(example, tokenizer, image_processor), 
        batched=False, 
        remove_columns=hf_dataset.column_names,
        load_from_cache_file=False 
    ).filter(lambda x: x is not None)

    if len(processed_dataset) == 0:
        raise ValueError("Processed dataset is empty. Ensure 'satellite.json' is valid and images exist.")

    print("\nMultimodal Model Trainable Parameters:")
    custom_model.language_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./llava_deepseek_output",
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="paged_adamw_8bit",
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False, 
    )
    
    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=MultimodalDataCollator(
            tokenizer=tokenizer, 
            image_processor=image_processor
        ),
    )
    
    print("\nStarting Multimodal Fine-Tuning...")
    trainer.train()

    custom_model.language_model.save_pretrained("./llava_deepseek_lora_adapters")
    torch.save(custom_model.mm_projector.state_dict(), "./llava_deepseek_projector.pt")
    print("Training Complete. Saved LoRA adapters and MLP projector.")