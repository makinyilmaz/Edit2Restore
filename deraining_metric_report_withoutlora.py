import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import numpy as np
from tqdm import tqdm
from CMMD.main import cmmd_func


try:
    from torchmetrics.image import NaturalImageQualityEvaluator
except ImportError:
    print("Warning: NIQE not available in your torchmetrics version")
    NaturalImageQualityEvaluator = None

try:
    from torchmetrics.multimodal import CLIPImageQualityAssessment
except ImportError:
    print("Warning: CLIPIQA not available in your torchmetrics version")
    CLIPImageQualityAssessment = None


def load_image(path, device):
    """Load an image and convert to tensor."""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def main():
    # Paths
    pred_folder = "results/deraining/Rain100L/withoutlora"
    gt_folder = "test-data/Rain100L/target"

    cmmd_score = cmmd_func(gt_folder, pred_folder)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    # Optional metrics
    niqe_metric = NaturalImageQualityEvaluator().to(device) if NaturalImageQualityEvaluator else None
    clipiqa_metric = CLIPImageQualityAssessment().to(device) if CLIPImageQualityAssessment else None
    
    # Get list of images
    pred_images = sorted([f for f in os.listdir(pred_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(pred_images)} images to evaluate")
    
    # Lists to store individual metric values
    psnr_values = []
    ssim_values = []
    niqe_values = []
    clipiqa_values = []
    
    # Process images
    for img_name in tqdm(pred_images, desc="Processing images"):
        pred_path = os.path.join(pred_folder, img_name)
        gt_path = os.path.join(gt_folder, img_name)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {img_name}, skipping...")
            continue
        
        # Load images
        pred_img = load_image(pred_path, device)
        gt_img = load_image(gt_path, device)
        
        # Ensure images have the same size
        if pred_img.shape != gt_img.shape:
            print(f"Warning: Size mismatch for {img_name}, resizing prediction to match GT")
            pred_img = torch.nn.functional.interpolate(
                pred_img, 
                size=(gt_img.shape[2], gt_img.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Calculate PSNR
        psnr_val = psnr_metric(pred_img, gt_img)
        psnr_values.append(psnr_val.item())
        print(img_name, psnr_val.item())
        # Calculate SSIM
        ssim_val = ssim_metric(pred_img, gt_img)
        ssim_values.append(ssim_val.item())
        
        # Update FID (requires batch processing)
        # Convert to uint8 format for FID
        pred_img_uint8 = (pred_img * 255).clamp(0, 255).to(torch.uint8)
        gt_img_uint8 = (gt_img * 255).clamp(0, 255).to(torch.uint8)
        fid_metric.update(gt_img_uint8, real=True)
        fid_metric.update(pred_img_uint8, real=False)
        
        # Calculate NIQE (no-reference metric, calculated on predictions)
        if niqe_metric:
            niqe_val = niqe_metric(pred_img_uint8)
            niqe_values.append(niqe_val.item())
        
        # Calculate CLIPIQA (no-reference metric)
        if clipiqa_metric:
            clipiqa_val = clipiqa_metric(pred_img)
            clipiqa_values.append(clipiqa_val.item())
    
    # Calculate FID
    fid_score = fid_metric.compute()
    
    # Calculate averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_niqe = np.mean(niqe_values) if niqe_values else None
    avg_clipiqa = np.mean(clipiqa_values) if clipiqa_values else None
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of images evaluated: {len(psnr_values)}")
    print(f"\nAverage PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"CMMD Score: {cmmd_score:.4f}")
    
    if avg_niqe is not None:
        print(f"Average NIQE: {avg_niqe:.4f} (lower is better)")
    else:
        print("NIQE: Not available")
    
    if avg_clipiqa is not None:
        print(f"Average CLIPIQA: {avg_clipiqa:.4f}")
    else:
        print("CLIPIQA: Not available")
    
    
    # Save results to file
    with open("results/deraining/evaluation_results_withoutlora.txt", "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Number of images evaluated: {len(psnr_values)}\n\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"CMMD Score: {cmmd_score:.4f}\n")
        if avg_niqe is not None:
            f.write(f"Average NIQE: {avg_niqe:.4f} (lower is better)\n")
        if avg_clipiqa is not None:
            f.write(f"Average CLIPIQA: {avg_clipiqa:.4f}\n")
    
    print("\nResults saved to 'results/deraining/evaluation_results_withoutlora.txt'")

if __name__ == "__main__":
    main()