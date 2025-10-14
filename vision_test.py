# vision_test.py
import torch
from PIL import Image
import open_clip

print("Loading CLIP Vision Model...")
# Load the specific pre-trained CLIP model we plan to use (ViT-L/14)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
model.to("cuda") # Move the model to your GPU
print("CLIP model loaded successfully.")

# Find a sample satellite image online and save it as 'sample_image.jpg' in your project folder
try:
    image = Image.open("sample_image.jpg")
    print("Sample image opened.")
except FileNotFoundError:
    raise SystemExit("Error: 'sample_image.jpg' not found. Please download a sample image to your project directory.")

# Preprocess the image to fit the model's requirements
image_input = preprocess(image).unsqueeze(0).to("cuda")

# Generate the image embedding
with torch.no_grad():
    image_features = model.encode_image(image_input)

print(f"Successfully generated image embedding!")
print(f"Shape of the embedding vector: {image_features.shape}") # Should be [1, 768]
