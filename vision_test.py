import torch
from PIL import Image
import open_clip

torch.backends.cudnn.benchmark = True

print("Loading CLIP Vision Model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
model.to("cuda")

print("CLIP model loaded successfully.")

try:
    image = Image.open("sample_image.jpg")
    print("Sample image opened.")
except FileNotFoundError:
    raise SystemExit("Error: 'sample_image.jpg' not found. Please place your EO image in the project directory.")

image_input = preprocess(image).unsqueeze(0).to("cuda")

with torch.no_grad():
    image_features = model.encode_image(image_input)
    print(f"Image embedding shape: {image_features.shape}")  # Should be [1, 768]

    # Optional: Multimodal consistency check, encode a sample text
    text = "Flooded region in satellite photo."
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    text_input = tokenizer([text])
    text_features = model.encode_text(text_input.to("cuda"))
    print(f"Text embedding shape: {text_features.shape}")  # Should be [1, 768]
