import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from pathlib import Path

# Initialize pipeline once
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", 
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(
    "lora_model128/pytorch_lora_weights.safetensors"
)
pipe.to("cuda")

# Define paths
input_dir = Path("test-data/Rain100L/input")
output_dir = Path("results/deraining/Rain100L/withlora")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Get all image files (common image extensions)
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
image_files = []
for ext in image_extensions:
    image_files.extend(input_dir.glob(ext))

print(f"Found {len(image_files)} images to process")

# Process each image
for img_path in image_files:
    print(f"Processing {img_path.name}...")
    
    # Load and convert image
    input_image = Image.open(img_path).convert("RGB")
    
    # Store original dimensions
    original_size = input_image.size  # (width, height)
    
    # Resize to 1024x1024
    input_image_resized = input_image.resize((1024, 1024), Image.LANCZOS)
    
    # Process the image
    image = pipe(
        image=input_image_resized,
        prompt="remove the rain from the image",
        guidance_scale=2.5,
    ).images[0]
    
    # Resize back to original dimensions
    image_original_size = image.resize(original_size, Image.LANCZOS)
    
    # Save with the same filename
    output_path = output_dir / img_path.name
    image_original_size.save(output_path)
    print(f"Saved to {output_path}")

print("All images processed!")