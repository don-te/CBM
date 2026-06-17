import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import warnings
from transformers import pipeline

# Suppress annoying warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
IMAGE_PATH = "/home/thejastg/sentinel_project/sentinel/palwal/rgb/27.9678_77.2410.png" # Make sure your blurry kiln patch is here

# The Macro-Concepts for our Concept Bottleneck Model (CBM)
PROMPTS = [
    "large industrial roof",
    "heavy shadow cast",
    "smoke plume",
    "cleared barren land",
    "brick kiln"
]
UPSCALED_SIZE = 1024 
CONFIDENCE_THRESHOLD = 0.03 # Lowered to 3% to catch uncertain top-down predictions

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"CRITICAL ERROR: Please place a test image at {IMAGE_PATH}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CRITICAL ERROR: PyTorch is still using your CPU! Fix your CUDA installation.")
        sys.exit(1)

    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print("Loading Google OWLv2 Zero-Shot Detector... ")
    
    detector = pipeline(
        model="google/owlv2-base-patch16-ensemble", 
        task="zero-shot-object-detection", 
        device=0 
    )

    original_pil = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = original_pil.size
    
    print(f"Upscaling image from {orig_w}x{orig_h} to {UPSCALED_SIZE}x{UPSCALED_SIZE} for processing...")
    image_pil = original_pil.resize((UPSCALED_SIZE, UPSCALED_SIZE), Image.Resampling.LANCZOS)
    
    scale_x = orig_w / float(UPSCALED_SIZE)
    scale_y = orig_h / float(UPSCALED_SIZE)
    
    fig, axes = plt.subplots(len(PROMPTS), 2, figsize=(12, 5 * len(PROMPTS)))
    if len(PROMPTS) == 1:
        axes = [axes] 
        
    for idx, prompt in enumerate(PROMPTS):
        print(f"Testing macro-concept: '{prompt}'...")
        
        results = detector(
            image_pil, 
            candidate_labels=[prompt],
            threshold=CONFIDENCE_THRESHOLD
        )
        
        ax_orig = axes[idx][0]
        ax_det = axes[idx][1]
        
        ax_orig.imshow(original_pil)
        ax_orig.set_title(f"Original Image")
        ax_orig.axis('off')
        
        ax_det.imshow(original_pil)
        ax_det.set_title(f"Prompt: '{prompt}'")
        ax_det.axis('off')
        
        if len(results) == 0:
            ax_det.text(0.5, 0.5, "NO DETECTIONS", color='red', fontsize=14, 
                         ha='center', va='center', transform=ax_det.transAxes)
            continue
            
        for detection in results:
            box = detection['box']
            score = detection['score']
            
            x_min = box['xmin'] * scale_x
            y_min = box['ymin'] * scale_y
            x_max = box['xmax'] * scale_x
            y_max = box['ymax'] * scale_y
            
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax_det.add_patch(rect)
            
            ax_det.text(x_min, y_min - 2, f"{score:.2f}", color='white', 
                        fontsize=10, backgroundcolor='red', fontweight='bold')

    plt.tight_layout()
    output_filename = "macro_concept_results.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nSUCCESS: Results saved to {output_filename}. Open this file to analyze the CBM triggers.")

if __name__ == "__main__":
    main()