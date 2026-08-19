import os
import torch
from PIL import Image
from diffusers import FluxKontextPipeline

input_dir = "test_data/denoising_testsets/CBSD68_25"
output_dir = "results/kontextlora_s32_r64_tenc_allinone/denoising_25"
os.makedirs(output_dir, exist_ok=True)

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(
    "trained_loras/fluxkontextdev/all-in-one/sample32_rank64_textencoder/pytorch_lora_weights.safetensors"
)
pipe.enable_model_cpu_offload()

prompt = "Remove noise and grain, restore a clean and sharp image"
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

files = sorted([
    f for f in os.listdir(input_dir)
    if os.path.splitext(f)[1].lower() in exts
])

print(f"Found {len(files)} images in {input_dir}")

for i, fname in enumerate(files, 1):
    print(f"[{i}/{len(files)}] Processing {fname}...")
    img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
    original_size = img.size

    # 1) Force resize to 1024x1024
    resized = img.resize((1024, 1024), Image.LANCZOS)

    # 2) Run Flux Kontext edit
    result = pipe(
        image=resized,
        prompt=prompt,
        guidance_scale=2.5,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    # 3) Resize back to original resolution
    result = result.resize(original_size, Image.LANCZOS)

    out_path = os.path.join(output_dir, fname)
    result.save(out_path)
    print(f"  Saved -> {out_path} ({original_size[0]}x{original_size[1]})")

print("Done!")