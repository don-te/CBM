import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import warnings

# Use native Hugging Face for both models
from transformers import pipeline, SamModel, SamProcessor

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONFIG = {
    "IMAGE_DIR": Path("/home/rishabh.mondal/mbzuai/cbm data/power_plants/images_y_2026_z_18_4096_still_factory/crops/"),
    "JSON_DIR": Path("/home/rishabh.mondal/mbzuai/cbm data/power_plants/images_y_2026_z_18_4096_still_factory/crops_pseudo_label/"),
    "OUTPUT_DIR": Path("./data/crops/segmentation_outputs_v2_structural/"),
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # DIRECTORY TARGETING CONTROLS
    "TARGET_FOLDERS": ["20.892166_84.992334_2026"], # Your exact target folder!
    "MAX_TEST_RUN": 5,
    
    # STRATEGY THRESHOLDS
    "OWL_CONFIDENCE_THRESHOLD": 0.16,   # Raised to kill weak shadow detections
    "MAX_BOX_AREA_RATIO": 0.40,         # Reject boxes larger than 40% of the image
    "NMS_IOU_THRESHOLD": 0.15,          # AGGRESSIVE: Kill overlaps greater than 15%
    "MAX_INSTANCES_PER_CLASS": 3        # ONLY keep the Top 3 best boxes per concept
}

# STRATEGY D: Macro to Micro Translation Dictionary (Morphological Prompting)
# Grouping functional JSON labels into purely structural, geometric buckets.
CONCEPT_TRANSLATION = {
    # --- BUCKET 1: Long, thin, linear structures ---
    "Railway Siding": "long straight steel train tracks",
    "Quarry Conveyor": "long straight elevated conveyor belt",
    
    # --- BUCKET 2: Large rectangular roofs ---
    "Industrial Building": "large flat rectangular building roof",
    "Boiler Building": "large flat rectangular building roof",
    "Rotary Kiln Building": "large flat rectangular building roof",
    
    # --- BUCKET 3: Solid circular geometric shapes (Top-Down) ---
    "Storage Tank": "bright white circular storage tank",
    "Cement Silo": "bright white circular storage tank",
    
    # --- BUCKET 4: Tall structures casting long shadows ---
    "Chimney Stack": "tall circular smokestack casting long shadow",
    "Kiln Chimney": "tall circular smokestack casting long shadow",
    "Cement Plant Chimney": "tall circular smokestack casting long shadow",
    
    # --- BUCKET 5: Massive hollow circular concrete ---
    "Cooling Tower": "massive hollow circular concrete cooling tower",
    
    # --- BUCKET 6: Dense textured grids ---
    "Switchyard": "dense grid of electrical transformers",
    
    # --- BUCKET 7: Dark amorphous pools ---
    "Water Body": "dark irregular water pond",
    "Wastewater Pond": "dark irregular water pond",
    
    # --- BUCKET 8: Textured piles ---
    "Coal Stockpile": "large pile of black coal"
}

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# --- PIPELINE FUNCTIONS ---

def setup_directories():
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    (CONFIG["OUTPUT_DIR"] / "visualizations").mkdir(exist_ok=True)
    (CONFIG["OUTPUT_DIR"] / "binary_masks").mkdir(exist_ok=True)

def load_models() -> Tuple[Any, SamModel, SamProcessor]:
    if CONFIG["DEVICE"] == "cpu":
        logger.error("PyTorch is on CPU! Please activate your CUDA environment.")
        sys.exit(1)
        
    logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    logger.info("Loading Google OWLv2 Detector...")
    detector = pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble", device=0)

    logger.info("Loading Meta SAM (Segment Anything)...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(CONFIG["DEVICE"])
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    return detector, sam_model, sam_processor

def parse_present_concepts(json_path: Path) -> List[str]:
    """Reads JSON, extracts 'Present' items, and EXCLUDES background noise."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    excluded_concepts = {"Bare Soil", "Paved Road", "Kiln Access Road", "Slag Dump"}
    return [c for c, d in data.items() if d.get("status") == "Present" and c not in excluded_concepts]

def run_owlv2(detector, image_pil: Image.Image, concepts: List[str]) -> Tuple[List[List[float]], List[str]]:
    """Runs OWLv2 with Translation, Area Filtering, and NMS Deduplication."""
    # 1. Translate concepts to visual prompts
    visual_prompts = [CONCEPT_TRANSLATION.get(c, c) for c in concepts]
    reverse_map = {CONCEPT_TRANSLATION.get(c, c): c for c in concepts} # To restore original JSON labels
    
    owl_results = detector(image_pil, candidate_labels=visual_prompts, threshold=CONFIG["OWL_CONFIDENCE_THRESHOLD"])
    
    raw_boxes, raw_labels, raw_scores = [], [], []
    img_w, img_h = image_pil.size
    img_area = img_w * img_h
    
    for res in owl_results:
        xmin, ymin, xmax, ymax = res["box"]["xmin"], res["box"]["ymin"], res["box"]["xmax"], res["box"]["ymax"]
        box_area = (xmax - xmin) * (ymax - ymin)
        
        # STRATEGY A: Area Threshold Filter (Kill massive bleeding boxes)
        if box_area / img_area > CONFIG["MAX_BOX_AREA_RATIO"]:
            continue
            
        raw_boxes.append([xmin, ymin, xmax, ymax])
        raw_labels.append(reverse_map.get(res["label"], res["label"]))
        raw_scores.append(res["score"])
        
    if not raw_boxes:
        return [], []
        
    # STRATEGY C: Deduplication (Non-Maximum Suppression)
    final_boxes, final_labels = [], []
    unique_labels = set(raw_labels)
    
    for label in unique_labels:
        # Isolate boxes and scores for this specific class
        idxs = [i for i, l in enumerate(raw_labels) if l == label]
        class_boxes = torch.tensor([raw_boxes[i] for i in idxs], dtype=torch.float32)
        class_scores = torch.tensor([raw_scores[i] for i in idxs], dtype=torch.float32)
        
        # PyTorch NMS
        keep_idxs = torchvision.ops.nms(class_boxes, class_scores, CONFIG["NMS_IOU_THRESHOLD"])
        
        # Top-K Filtering
        sorted_keep_idxs = sorted(keep_idxs.tolist(), key=lambda idx: class_scores[idx].item(), reverse=True)
        top_k_idxs = sorted_keep_idxs[:CONFIG["MAX_INSTANCES_PER_CLASS"]]
        
        for keep_i in top_k_idxs:
            original_i = idxs[keep_i]
            final_boxes.append(raw_boxes[original_i])
            final_labels.append(raw_labels[original_i])
            
    return final_boxes, final_labels

def run_sam(sam_model, sam_processor, image_pil: Image.Image, input_boxes: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feeds Strict Bounding Boxes to SAM."""
    inputs = sam_processor(image_pil, input_boxes=[input_boxes], return_tensors="pt").to(CONFIG["DEVICE"])
    
    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks_tensor = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0] 
    
    iou_scores = outputs.iou_scores[0].cpu()
    return masks_tensor, iou_scores

def export_results(base_name, folder_name, image_pil, input_boxes, labels, masks_tensor, iou_scores):
    """Draws a 1x2 side-by-side comparison visualization."""
    img_w, img_h = image_pil.size
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- LEFT SIDE: Original Clean Image ---
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Crop", fontsize=16, fontweight='bold', color='black')
    axes[0].axis('off')
    
    # --- RIGHT SIDE: Annotated Output ---
    axes[1].imshow(image_pil)
    axes[1].set_title("AI Segmentation (OWLv2 + SAM)", fontsize=16, fontweight='bold', color='black')
    axes[1].axis('off')
    
    combined_binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    color_map = plt.get_cmap('hsv', max(1, len(input_boxes)))

    for i, box in enumerate(input_boxes):
        best_mask_idx = torch.argmax(iou_scores[i]).item()
        best_mask = masks_tensor[i, best_mask_idx].numpy()
        color = color_map(i)[:3]

        combined_binary_mask[best_mask] = 255
        
        colored_mask = np.zeros((img_h, img_w, 4))
        colored_mask[best_mask] = [*color, 0.5] 
        
        axes[1].imshow(colored_mask)
        
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        axes[1].add_patch(rect)
        axes[1].text(xmin, ymin - 5, labels[i], color='white', fontsize=10, bbox=dict(facecolor=color, edgecolor='none', alpha=0.8))

    vis_path = CONFIG["OUTPUT_DIR"] / "visualizations" / folder_name / f"seg_vis_{base_name}.png"
    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)

    mask_path = CONFIG["OUTPUT_DIR"] / "binary_masks" / folder_name / f"mask_{base_name}.png"
    cv2.imwrite(str(mask_path), combined_binary_mask)

def process_single_image(json_path: Path, folder_name: str, detector, sam_model, sam_processor) -> bool:
    base_name = json_path.stem
    img_path = CONFIG["IMAGE_DIR"] / folder_name / f"{base_name}.png"
    if not img_path.exists():
        img_path = CONFIG["IMAGE_DIR"] / folder_name / f"{base_name}.jpg"
        if not img_path.exists():
            return False

    present_concepts = parse_present_concepts(json_path)
    if not present_concepts:
        return False

    image_pil = Image.open(img_path).convert("RGB")

    input_boxes, labels = run_owlv2(detector, image_pil, present_concepts)
    if not input_boxes:
        return False

    masks_tensor, iou_scores = run_sam(sam_model, sam_processor, image_pil, input_boxes)
    export_results(base_name, folder_name, image_pil, input_boxes, labels, masks_tensor, iou_scores)
    return True

def main():
    setup_directories()
    detector, sam_model, sam_processor = load_models()
        
    json_folders = sorted([d for d in CONFIG["JSON_DIR"].iterdir() if d.is_dir()])
    
    # DIRECTORY TARGETING LOGIC
    if CONFIG.get("TARGET_FOLDERS"):
        target_set = set(CONFIG["TARGET_FOLDERS"])
        json_folders = [d for d in json_folders if d.name in target_set]
        logger.info(f"TARGET_FOLDERS enabled: Restricting run strictly to {len(json_folders)} specified folders.")
    elif CONFIG.get("MAX_TEST_RUN") is not None:
        logger.info(f"TRIAL RUN ENABLED: Limiting to {CONFIG['MAX_TEST_RUN']} lat_lon folders.")
        json_folders = json_folders[:CONFIG["MAX_TEST_RUN"]]

    if not json_folders:
        logger.error("No valid folders found to process. Check your TARGET_FOLDERS names or directory paths.")
        sys.exit(1)

    success_count = 0

    for folder_idx, folder_path in enumerate(json_folders, 1):
        folder_name = folder_path.name
        logger.info(f"=== Processing Folder {folder_idx}/{len(json_folders)}: {folder_name} ===")
        
        json_files = list(folder_path.glob("*.json"))
        (CONFIG["OUTPUT_DIR"] / "visualizations" / folder_name).mkdir(parents=True, exist_ok=True)
        (CONFIG["OUTPUT_DIR"] / "binary_masks" / folder_name).mkdir(parents=True, exist_ok=True)

        for json_path in json_files:
            try:
                if process_single_image(json_path, folder_name, detector, sam_model, sam_processor):
                    success_count += 1
            except Exception as e:
                logger.error(f"  -> [{json_path.stem}] FATAL ERROR: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info(f"BATCH COMPLETE: Successfully segmented {success_count} crops.")

if __name__ == "__main__":
    main()