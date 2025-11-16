Fix description: this script is for SAR ship LoRA fine-tuning, not generic COCO


import os
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_coco(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    anns_by_img: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    return images, anns_by_img, cats


def find_lora_targets(model) -> List[str]:
    patterns = [r"\.q_proj$", r"\.k_proj$", r"\.v_proj$", r"\.out_proj$"]
    names = set()
    for name, _module in model.named_modules():
        if any(re.search(p, name) for p in patterns):
            names.add(name.split(".")[-1])
    found = sorted(list(names))
    print("[LoRA targets(found)]:", found if found else ["q_proj", "k_proj", "v_proj", "out_proj"])
    return found if found else ["q_proj", "k_proj", "v_proj", "out_proj"]


class GdinoCocoDS(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        anns_by_img,
        img_dir,
        processor,
        text_prompt,
        catid_to_prompt_idx,
    ):
        self.images = images
        self.anns_by_img = anns_by_img
        self.img_dir = img_dir
        self.processor = processor
        self.text_prompt = text_prompt
        self.catid_to_prompt_idx = catid_to_prompt_idx
        self.ids = list(images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> Dict[str, Any]:
        img_id = self.ids[idx]
        info = self.images[img_id]
        path = os.path.join(self.img_dir, info["file_name"])
        image = Image.open(path).convert("RGB")

        raw_anns = self.anns_by_img.get(img_id, [])
        anns = []
        for a in raw_anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cid = a["category_id"]
            if cid not in self.catid_to_prompt_idx:
                continue
            new_cid = self.catid_to_prompt_idx[cid]
            anns.append(
                {
                    "bbox": [x, y, w, h],
                    "category_id": new_cid,
                    "iscrowd": a.get("iscrowd", 0),
                    "area": a.get("area", w * h),
                }
            )

        ann_dict = {"image_id": img_id, "annotations": anns}

        out = self.processor(
            images=image,
            text=self.text_prompt,
            annotations=ann_dict,
            return_tensors="pt",
        )
        return {k: v[0] for k, v in out.items()}


@dataclass
class GdinoCollator:
    processor: AutoProcessor

    def __call__(self, features):
        pixel_values = [f["pixel_values"] for f in features]
        pixel_values = self.processor.image_processor.pad(
            pixel_values, return_tensors="pt"
        )["pixel_values"]

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        text = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt",
        )

        labels = [f["labels"] for f in features]

        return {
            "pixel_values": pixel_values,
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "labels": labels,
        }


def train_gdino_lora(
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    train_images: str = None,
    train_ann: str = None,
    val_images: str = None,
    val_ann: str = None,
    save_dir: str = "./checkpoints/gdino_lora",
    epochs: int = 5,
    lr: float = 1e-4,
    batch: int = 2,
    grad_accum: int = 1,
    seed: int = 42,
    fp16: bool = True,
    save_pth: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_targets: List[str] = None,
    num_workers: int = 0,
):
    assert train_images and train_ann and val_images and val_ann, "Missing dataset paths."
    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_images, tr_anns_by_img, cats = load_coco(train_ann)
    va_images, va_anns_by_img, _ = load_coco(val_ann)

    cat_names = sorted(set(cats.values()))
    text_prompt = ". ".join(cat_names) + "."
    print(f"[INFO] categories: {cat_names}")
    print(f"[INFO] prompt: {text_prompt}")

    name_to_idx = {name: i for i, name in enumerate(cat_names)}
    catid_to_prompt_idx = {cid: name_to_idx[name] for cid, name in cats.items()}
    print("[INFO] catid_to_prompt_idx:", catid_to_prompt_idx)

    def _uniq_ids(anns_by_img):
        return sorted({a["category_id"] for anns in anns_by_img.values() for a in anns})

    print("[DEBUG] unique train category_id:", _uniq_ids(tr_anns_by_img))
    print("[DEBUG] unique valid category_id:", _uniq_ids(va_anns_by_img))

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.config.use_cache = False
    model.train()

    if getattr(model, "supports_gradient_checkpointing", False) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    else:
        print("[INFO] Gradient checkpointing not supported; skipping.")

    if lora_targets is None:
        lora_targets = find_lora_targets(model)

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = GdinoCocoDS(
        tr_images,
        tr_anns_by_img,
        train_images,
        processor,
        text_prompt,
        catid_to_prompt_idx,
    )
    val_ds = GdinoCocoDS(
        va_images,
        va_anns_by_img,
        val_images,
        processor,
        text_prompt,
        catid_to_prompt_idx,
    )
    collator = GdinoCollator(processor)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        fp16=fp16 and torch.cuda.is_available(),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        dataloader_num_workers=num_workers,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    adapter_dir = os.path.join(save_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    if save_pth:
        sd_path = os.path.join(save_dir, "model_lora_state_dict.pth")
        torch.save(model.state_dict(), sd_path)
        print(f"state_dict saved to: {sd_path}")

    print("Done fine-tuning.")
    return adapter_dir


def load_for_inference(model_id: str, adapter_dir: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    processor = AutoProcessor.from_pretrained(adapter_dir)
    return model, processor


if __name__ == "__main__":
    adapter_path = train_gdino_lora(
        model_id="IDEA-Research/grounding-dino-tiny",
        train_images="path/to/train/images",
        train_ann="path/to/train/_annotations.coco.json",
        val_images="path/to/valid/images",
        val_ann="path/to/valid/_annotations.coco.json",
        save_dir="./checkpoints/gdino_lora",
        epochs=10,
        batch=2,
        lr=1e-4,
        fp16=True,
        save_pth=True,
        grad_accum=1,
    )
