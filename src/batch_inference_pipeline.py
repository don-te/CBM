import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import pipeline
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_DIR = "/home/thejastg/sentinel_project/sentinel/palwal/rgb"  # Folder containing your Sentinel-2 tiles
OUTPUT_DIR = "./data/cbm_batch_results/" # Where the visual proofs and charts will go

# Mix of Micro (will likely fail) and Macro (will likely succeed) concepts
PROMPTS = [
    "industrial chimney",    # Micro
    "tall smokestack",       # Micro
    "heavy shadow cast",     # Macro
    "large industrial roof", # Macro
    "cleared barren land"    # Macro
]

UPSCALED_SIZE = 1024 
CONFIDENCE_THRESHOLD = 0.03 # 3% threshold for top-down satellite imagery

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"CRITICAL ERROR: Input directory {INPUT_DIR} does not exist.")
        sys.exit(1)
        
    # Setup output directories
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"CRITICAL ERROR: No images found in {INPUT_DIR}.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CRITICAL ERROR: PyTorch is still using your CPU! Fix your CUDA installation.")
        sys.exit(1)

    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    print("Loading Google OWLv2 Zero-Shot Detector... (Native Hugging Face Pipeline)")
    
    detector = pipeline(
        model="google/owlv2-base-patch16-ensemble", 
        task="zero-shot-object-detection", 
        device=0 
    )

    print(f"Starting batch CBM extraction on {len(image_files)} images...")
    
    # Tracking metrics
    stats = {prompt: 0 for prompt in PROMPTS}
    all_detections = []
    processed_count = 0

    for filename in image_files:
        input_path = os.path.join(INPUT_DIR, filename)
        
        try:
            original_pil = Image.open(input_path).convert("RGB")
            orig_w, orig_h = original_pil.size
        except Exception as e:
            print(f"Skipping {filename}: Could not read image. ({e})")
            continue
            
        # Upscale to prevent feature collapse
        image_pil = original_pil.resize((UPSCALED_SIZE, UPSCALED_SIZE), Image.Resampling.LANCZOS)
        scale_x = orig_w / float(UPSCALED_SIZE)
        scale_y = orig_h / float(UPSCALED_SIZE)
        
        # Run inference on all prompts at once
        results = detector(
            image_pil, 
            candidate_labels=PROMPTS,
            threshold=CONFIDENCE_THRESHOLD
        )
        
        # Track if we need to save a visualization for this image
        drawn_boxes = False
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(original_pil)
        ax.axis('off')
        
        # Process results
        for detection in results:
            label = detection['label']
            score = detection['score']
            box = detection['box']
            
            # Log the hit
            stats[label] += 1
            all_detections.append({
                "image": filename,
                "concept": label,
                "confidence": round(score, 4),
                "xmin": box['xmin'] * scale_x,
                "ymin": box['ymin'] * scale_y,
                "xmax": box['xmax'] * scale_x,
                "ymax": box['ymax'] * scale_y
            })
            
            # Draw bounding box for visualization
            x_min, y_min = box['xmin'] * scale_x, box['ymin'] * scale_y
            width = (box['xmax'] - box['xmin']) * scale_x
            height = (box['ymax'] - box['ymin']) * scale_y
            
            # Color code: Red for Micro, Blue for Macro
            edge_color = 'red' if label in ["industrial chimney", "tall smokestack"] else 'cyan'
            
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=edge_color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 2, f"{label} ({score:.2f})", color='white', fontsize=8, backgroundcolor=edge_color)
            drawn_boxes = True
            
        if drawn_boxes:
            output_vis_path = os.path.join(vis_dir, f"annotated_{filename}")
            plt.savefig(output_vis_path, bbox_inches='tight', pad_inches=0, dpi=150)
            
        plt.close(fig) # Free up memory
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{len(image_files)} images...")

    # --- SAVE CSV LOG ---
    df = pd.DataFrame(all_detections)
    csv_path = os.path.join(OUTPUT_DIR, "cbm_detections_log.csv")
    df.to_csv(csv_path, index=False)

    # --- GENERATE PROOF CHART FOR PROFESSOR ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stats.keys(), stats.values(), color=['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#3498db'])
    plt.title(f"Concept Bottleneck Detection Rates\n(Total Images: {len(image_files)} | Sentinel-2 10m Resolution)", fontsize=14)
    plt.ylabel("Total Successful Detections", fontsize=12)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    
    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#e74c3c', lw=4), Line2D([0], [0], color='#3498db', lw=4)]
    plt.legend(custom_lines, ['Micro Concepts (Domain Gap / Sub-pixel)', 'Macro Concepts (Primitive Features)'])
    
    chart_path = os.path.join(OUTPUT_DIR, "professor_proof_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)

    print(f"\nSUCCESS: Batch inference complete!")
    print(f"-> Annotated interpretability images saved to: {vis_dir}")
    print(f"-> Raw data log saved to: {csv_path}")
    print(f"-> Professor's Proof Chart saved to: {chart_path}")

if __name__ == "__main__":
    main()