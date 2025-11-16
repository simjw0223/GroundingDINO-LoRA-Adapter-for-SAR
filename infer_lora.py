"""
==========================================================
LoRA-Finetuned Grounding DINO - Inference & COCO Evaluation
==========================================================

This script performs inference using a Grounding-DINO-Tiny model 
combined with a LoRA adapter trained on custom data (e.g., SAR ship detection).
It supports:

    ● Text-prompt-based zero-shot detection
    ● Visualization and bounding box overlay
    ● Exporting COCO-style prediction JSON
    ● COCO evaluation (mAP, AP50, AP75, AR)

Base Model : IDEA-Research / grounding-dino-tiny
LoRA Adapter Folder : exported during training phase
----------------------------------------------------------
"""

import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from peft import PeftModel

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# --------------------------------------------------------
# 0) Path Configuration
# --------------------------------------------------------
model_id = "IDEA-Research/grounding-dino-tiny"
adapter_dir = "./weights/lora_adapter"        # <--- replace with your LoRA adapter path
device = "cuda" if torch.cuda.is_available() else "cpu"

input_dir = "./data/test/images"              # input images
output_dir = "./results_lora"                 # visualization + JSON output
os.makedirs(output_dir, exist_ok=True)

gt_json_path = "./data/test/_annotations.coco.json"   # COCO ground-truth
pred_json_path = os.path.join(output_dir, "predictions_coco.json")

text_prompt = "ship."               # change for custom datasets
score_thresh = 0.30
text_threshold = 0.30
CATEGORY_ID = 1                     # for single-class datasets

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------
# 1) Load Base Model + LoRA Adapter
# --------------------------------------------------------
print("[INFO] Loading base model & LoRA adapter...")
base = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()

# Using processor exported with LoRA adapter ensures consistent preprocessing
processor = AutoProcessor.from_pretrained(adapter_dir)


# --------------------------------------------------------
# 2) Load COCO Dataset & Map Filenames
# --------------------------------------------------------
print("[INFO] Loading COCO annotations...")
coco_gt = COCO(gt_json_path)
fname_to_imgid = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}


# --------------------------------------------------------
# 3) Inference + Visualization + Prediction Logging
# --------------------------------------------------------
predictions = []

def clamp(v, lo, hi):
    return max(lo, min(float(v), hi))

print("[INFO] Starting inference...")

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(IMG_EXTS):
        continue

    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    image_id = fname_to_imgid.get(filename, None)

    # Preprocess
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        target_sizes=[(H, W)],
        text_threshold=text_threshold,
        text_labels=[[text_prompt]],
    )
    result = results[0]

    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, int(W * 0.015)))
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        score = float(score)
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = clamp(x1, 0, W - 1); y1 = clamp(y1, 0, H - 1)
        x2 = clamp(x2, 0, W - 1); y2 = clamp(y2, 0, H - 1)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        if image_id is not None:
            predictions.append({
                "image_id": int(image_id),
                "category_id": CATEGORY_ID,
                "bbox": [x1, y1, w, h],
                "score": score
            })

        label_text = f"{(label or 'ship').strip('.')} ({score:.3f})"
        draw.text((x1, y1 - 18), label_text, fill=(255, 255, 255, 255), font=font)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)

    image.save(os.path.join(output_dir, f"pred_{filename}"))

# Save JSON
with open(pred_json_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
print(f"[INFO] Predictions saved : {pred_json_path}")


# --------------------------------------------------------
# 4) COCO Evaluation
# --------------------------------------------------------
if len(predictions) == 0:
    print("[WARN] No valid predictions. Try lowering thresholds.")
else:
    print("[INFO] Running COCO evaluation...")
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    coco_eval.params.imgIds = list(fname_to_imgid.values())
    coco_eval.params.catIds = [CATEGORY_ID]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
