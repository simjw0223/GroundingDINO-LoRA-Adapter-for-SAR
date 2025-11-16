"""
LoRA-Based Inference & COCO Evaluation Script for Grounding DINO
-----------------------------------------------------------------

This script runs inference using a Grounding DINO model combined with
a LoRA adapter trained on any COCO-style dataset. It also supports
COCO evaluation metrics (mAP, AP50, AP75, AR) and can save per-image
visualization results along with a predictions JSON file.

Requirements:
    pip install transformers peft torch pillow pycocotools

Usage Example:
    python infer_groundingdino_lora.py \
        --base_model IDEA-Research/grounding-dino-tiny \
        --adapter_dir ./weights/lora_adapter \
        --input_dir ./dataset/test/images \
        --gt_json ./dataset/test/_annotations.coco.json \
        --output_dir ./results \
        --prompt "ship" \
        --score_thresh 0.3 \
        --text_thresh 0.3
"""

import os
import json
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from peft import PeftModel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def clamp(v, lo, hi):
    return max(lo, min(float(v), hi))


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load base + LoRA adapter
    base = AutoModelForZeroShotObjectDetection.from_pretrained(args.base_model).to(device).eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.adapter_dir)

    # Load GT COCO for evaluation
    print("Loading COCO ground-truth annotations...")
    coco_gt = COCO(args.gt_json)
    fname_to_imgid = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

    predictions = []

    print("\nRunning inference...")
    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith(IMG_EXTS):
            continue

        img_path = os.path.join(args.input_dir, filename)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        image_id = fname_to_imgid.get(filename, None)

        # Preprocess
        inputs = processor(images=image, text=args.prompt, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=[(h, w)],
            text_threshold=args.text_thresh,
            text_labels=[[args.prompt]],
        )[0]

        draw = ImageDraw.Draw(image, "RGBA")
        try:
            font = ImageFont.truetype("arial.ttf", size=max(16, int(w * 0.015)))
        except:
            font = ImageFont.load_default()

        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            score = float(score)
            if score < args.score_thresh:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            x1, y1 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1)
            x2, y2 = clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)

            if image_id is not None:
                predictions.append({
                    "image_id": int(image_id),
                    "category_id": args.category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                })

            clean_label = (label or args.prompt).strip(".")
            caption = f"{clean_label} ({score:.3f})"

            try:
                tb = draw.textbbox((0, 0), caption, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except:
                tw, th = int(draw.textlength(caption, font=font)), font.size + 6

            pad = 4
            y_label = max(0, y1 - th - pad)

            draw.rectangle([x1, y_label, x1 + tw + pad * 2, y_label + th + pad * 2], fill=(0, 0, 0, 160))
            draw.text((x1 + pad, y_label + pad), caption, fill=(255, 255, 255, 255), font=font)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)

        save_path = os.path.join(args.output_dir, f"pred_{filename}")
        image.save(save_path)
        print(f"[Saved] {save_path}")

    pred_json = os.path.join(args.output_dir, "predictions.json")
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved: {pred_json}")

    # Evaluation
    if predictions:
        print("\nRunning COCO evaluation...")
        coco_dt = coco_gt.loadRes(pred_json)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = list(fname_to_imgid.values())
        coco_eval.params.catIds = [args.category_id]
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    else:
        print("\n⚠️ No predictions above threshold. Adjust score/text thresholds.")


def main():
    parser = argparse.ArgumentParser(description="LoRA Inference & Evaluation for Grounding DINO")

    parser.add_argument("--base_model", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt (ex: 'ship')")
    parser.add_argument("--category_id", type=int, default=1)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--text_thresh", type=float, default=0.3)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
