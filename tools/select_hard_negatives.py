"""
Standalone Hard Negative Learning Pipeline for YOLO models.

Extracts hard negative / hard challenging samples from an initial dataset using a trained
detector (with SAHI sliced inference or full-image inference), evaluates model performance
per image, generates a hard-sample training dataset configuration, and fine-tunes the YOLO model.

Completely self-contained: requires only standard third-party libraries (ultralytics, sahi,
torch, torchmetrics, pandas, numpy, PIL, pyyaml, tqdm).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_hard_negatives")


try:
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    from torchmetrics.detection import MeanAveragePrecision
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO
    import torch
except ImportError as e:
    logger.error(f"Please install torchmetrics, sahi, ultralytics, torch, tqdm, numpy, pandas, pillow: {e}")
    sys.exit(1)





# ==========================================
# 1. YAML & Cache Helpers
# ==========================================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str, mode: str = "w") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def remove_label_cache(data_config_yaml: str) -> None:
    """Removes stale labels.cache files from YOLO datasets."""
    try:
        yolo_config = load_yaml(data_config_yaml)
        root = yolo_config.get("path", "")
        for split in ["train", "val", "test"]:
            split_paths = yolo_config.get(split)
            if not split_paths:
                continue
            if isinstance(split_paths, str):
                split_paths = [split_paths]
            for p in split_paths:
                full_path = os.path.join(root, p) if root else p
                # Handle directory or file
                candidate_cache_dirs = [
                    os.path.join(full_path, "../labels.cache"),
                    os.path.join(full_path, "labels.cache"),
                    Path(full_path).with_suffix(".cache"),
                ]
                for cache_file in candidate_cache_dirs:
                    if os.path.exists(cache_file):
                        try:
                            os.remove(cache_file)
                            logger.info(f"Removed cache: {cache_file}")
                        except Exception as e:
                            logger.warning(f"Could not remove {cache_file}: {e}")
    except Exception as e:
        logger.warning(f"Error while attempting to clear label cache: {e}")


# ==========================================
# 2. Image and Ground Truth Loading
# ==========================================

def get_label_path(image_path: str | Path) -> Path:
    """Resolves label .txt path from image path adhering to YOLO convention."""
    p_str = str(image_path)
    for sep in ["/", "\\"]:
        pattern = f"{sep}images{sep}"
        if pattern in p_str:
            parts = p_str.rsplit(pattern, 1)
            new_path_str = f"{sep}labels{sep}".join(parts)
            return Path(new_path_str).with_suffix(".txt")
    return Path(p_str.replace("images", "labels")).with_suffix(".txt")


def get_images_from_split(yolo_config: Dict[str, Any], split: str = "train", pattern_glob: str = "*") -> List[str]:
    """Retrieves all image paths for a given dataset split from a YOLO dataset YAML."""
    root = yolo_config.get("path", "")
    split_entry = yolo_config.get(split)
    if not split_entry:
        raise ValueError(f"Split '{split}' not defined in dataset config.")

    if isinstance(split_entry, str):
        split_paths = [split_entry]
    else:
        split_paths = split_entry

    image_paths: List[str] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    for entry in split_paths:
        full_entry_path = os.path.join(root, entry) if root and not os.path.isabs(entry) else entry
        p = Path(full_entry_path)

        if p.is_file():
            # If it is a text file with paths
            if p.suffix.lower() == ".txt":
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            if not os.path.isabs(line_str) and root:
                                line_str = str(Path(os.path.join(root, line_str)).resolve())
                            image_paths.append(line_str)
            elif p.suffix.lower() in valid_exts:
                image_paths.append(str(p.resolve()))
        elif p.is_dir():
            for f in p.glob(pattern_glob):
                if f.suffix.lower() in valid_exts:
                    image_paths.append(str(f.resolve()))

    return sorted(list(set(image_paths)))


def sample_pos_neg(images_paths: List[str], ratio: float, seed: int = 41) -> List[str]:
    """
    Subsamples negative (empty) images according to ratio = num_empty / num_non_empty.
    Preserves all non-empty (positive) images.
    """
    if len(images_paths) == 0:
        return []

    is_empty_list = []
    for p in images_paths:
        label_file = get_label_path(p)
        if not label_file.exists() or label_file.stat().st_size == 0:
            is_empty_list.append(1)
        else:
            is_empty_list.append(0)

    data = pd.DataFrame({"image_paths": images_paths, "is_empty": is_empty_list})
    num_empty = int((data["is_empty"] == 1).sum())
    num_non_empty = len(data) - num_empty

    if num_empty == 0:
        logger.info("Dataset contains only positive samples (no empty images found).")
        return data["image_paths"].tolist()

    num_sampled_empty = min(int(num_non_empty * ratio), num_empty)
    logger.info(
        f"Sampling images: {num_non_empty} positive, {num_sampled_empty} sampled empty "
        f"(out of {num_empty} total empty, ratio={ratio})."
    )

    sampled_empty = data.loc[data["is_empty"] == 1].sample(n=num_sampled_empty, random_state=seed)
    sampled_data = pd.concat([sampled_empty, data.loc[data["is_empty"] == 0]])
    return sampled_data["image_paths"].tolist()


def load_yolo_groundtruth(images_paths: List[str]) -> Tuple[pd.DataFrame, int]:
    """
    Loads YOLO format ground truth labels corresponding to the provided images.
    Converts normalized coordinates to pixel [x_min, y_min, x_max, y_max].
    """
    gt_records = []
    num_empty = 0

    for img_path_str in tqdm(images_paths, desc="Loading ground truth annotations"):
        img_path = Path(img_path_str)
        label_path = get_label_path(img_path)

        if not label_path.exists() or label_path.stat().st_size == 0:
            num_empty += 1
            continue

        try:
            df = pd.read_csv(label_path, sep=r"\s+", header=None)
            if df.empty or len(df.columns) < 5:
                num_empty += 1
                continue

            with Image.open(img_path) as img:
                w, h = img.size

            df = df.iloc[:, :5]
            df.columns = ["category_id", "x", "y", "w", "h"]

            # Convert YOLO normalized xywh to absolute xyxy
            x_min = (df["x"] - df["w"] / 2.0) * w
            x_max = (df["x"] + df["w"] / 2.0) * w
            y_min = (df["y"] - df["h"] / 2.0) * h
            y_max = (df["y"] + df["h"] / 2.0) * h

            for cat_id, x1, y1, x2, y2 in zip(df["category_id"], x_min, y_min, x_max, y_max):
                gt_records.append({
                    "file_name": str(img_path),
                    "category_id": int(cat_id),
                    "x_min": float(max(0.0, x1)),
                    "y_min": float(max(0.0, y1)),
                    "x_max": float(min(float(w), x2)),
                    "y_max": float(min(float(h), y2)),
                })
        except Exception as e:
            logger.warning(f"Failed to read ground truth for {label_path}: {e}")
            num_empty += 1

    df_gt = pd.DataFrame(gt_records)
    if df_gt.empty:
        df_gt = pd.DataFrame(columns=["file_name", "category_id", "x_min", "y_min", "x_max", "y_max"])
    return df_gt, num_empty


# ==========================================
# 3. Model Inference (SAHI or Native YOLO)
# ==========================================

class DetectorInference:
    def __init__(
        self,
        weights_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_sliding_window: bool = True,
        confidence_threshold: float = 0.25,
        overlap_ratio: float = 0.2,
        tilesize: int = 1280,
        imgsz: int = 1280,
    ):
        self.weights_path = weights_path
        self.device = device
        self.use_sliding_window = use_sliding_window
        self.confidence_threshold = confidence_threshold
        self.overlap_ratio = overlap_ratio
        self.tilesize = tilesize
        self.imgsz = imgsz

        if self.use_sliding_window:
            logger.info(f"Initializing SAHI detection model with {weights_path} on {device}")
            self.sahi_model = UltralyticsDetectionModel(
                model=YOLO(weights_path, task="detect"),
                confidence_threshold=confidence_threshold,
                image_size=imgsz,
                device=device,
            )
        else:
            logger.info(f"Initializing standard YOLO detection model with {weights_path} on {device}")
            self.yolo_model = YOLO(weights_path, task="detect")

    def predict_images(self, images_paths: List[str]) -> pd.DataFrame:
        """Runs predictions over a list of images and returns a unified DataFrame."""
        records = []

        with torch.no_grad():
            for img_path_str in tqdm(images_paths, desc="Running detector inference"):
                if not os.path.exists(img_path_str):
                    continue

            if self.use_sliding_window:
                try:
                    with Image.open(img_path_str) as img:
                        sliced_pred = get_sliced_prediction(
                            img,
                            self.sahi_model,
                            slice_height=self.tilesize,
                            slice_width=self.tilesize,
                            overlap_height_ratio=self.overlap_ratio,
                            overlap_width_ratio=self.overlap_ratio,
                            postprocess_type="NMS",
                            postprocess_match_metric="IOU",
                            postprocess_match_threshold=0.5,
                            verbose=0,
                        )
                        for obj in sliced_pred.object_prediction_list:
                            records.append({
                                "file_name": img_path_str,
                                "category_id": int(obj.category.id),
                                "score": float(obj.score.value),
                                "x_min": float(obj.bbox.minx),
                                "y_min": float(obj.bbox.miny),
                                "x_max": float(obj.bbox.maxx),
                                "y_max": float(obj.bbox.maxy),
                            })
                except Exception as e:
                    logger.warning(f"SAHI prediction failed on {img_path_str}: {e}")
            else:
                try:
                    results = self.yolo_model.predict(
                        source=img_path_str,
                        conf=self.confidence_threshold,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                    for r in results:
                        boxes = r.boxes
                        if boxes is None or len(boxes) == 0:
                            continue
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        for box, conf, cls_id in zip(xyxy, confs, clss):
                            records.append({
                                "file_name": img_path_str,
                                "category_id": int(cls_id),
                                "score": float(conf),
                                "x_min": float(box[0]),
                                "y_min": float(box[1]),
                                "x_max": float(box[2]),
                                "y_max": float(box[3]),
                            })
                except Exception as e:
                    logger.warning(f"YOLO prediction failed on {img_path_str}: {e}")

        df_pred = pd.DataFrame(records)
        if df_pred.empty:
            df_pred = pd.DataFrame(columns=["file_name", "category_id", "score", "x_min", "y_min", "x_max", "y_max"])
        return df_pred


# ==========================================
# 4. Evaluation & Hard Negative Selection
# ==========================================

def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """Computes standard pairwise IoU matrix between box1 (N, 4) and box2 (M, 4)."""
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    x11, y11, x12, y12 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
    x21, y21, x22, y22 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(x11, x21)
    inter_y1 = np.maximum(y11, y21)
    inter_x2 = np.minimum(x12, x22)
    inter_y2 = np.minimum(y12, y22)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    area2 = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-8)


def evaluate_image_metrics(
    candidate_images: List[str],
    df_pred: pd.DataFrame,
    df_gt: pd.DataFrame,
    tp_iou_threshold: float = 0.4,
    uncertainty_method: str = "entropy",
    single_cls: bool = False,
) -> pd.DataFrame:
    """
    Computes per-image metrics: True Positives (TP), False Positives (FP), False Negatives (FN),
    mAP50, mAP75, max_score, and uncertainty.
    """
    per_img_records = []

    m_ap = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5, 0.75],
    )

    for img_path in tqdm(candidate_images, desc="Evaluating detection performance"):
        pred_i = df_pred.loc[df_pred["file_name"] == img_path]
        gt_i = df_gt.loc[df_gt["file_name"] == img_path]

        num_preds = len(pred_i)
        num_gts = len(gt_i)

        scores = pred_i["score"].to_numpy() if num_preds > 0 else np.array([], dtype=np.float32)
        max_score = float(scores.max()) if num_preds > 0 else 0.0

        # Uncertainty
        if len(scores) > 0:
            if uncertainty_method == "entropy":
                clipped_scores = np.clip(scores, 1e-12, 1.0)
                uncertainty = float(-1.0 * np.sum(np.log(clipped_scores) * clipped_scores))
            else:  # '1-p'
                uncertainty = float(1.0 - np.mean(scores))
        else:
            uncertainty = 0.0

        # mAP calculation (reset metric state per image to prevent cumulative accumulation)
        map50, map75 = 0.0, 0.0
        if num_preds > 0 and num_gts > 0 and m_ap is not None:
            try:
                m_ap.reset()
                preds_tensor = [{
                    "boxes": torch.from_numpy(pred_i[["x_min", "y_min", "x_max", "y_max"]].to_numpy().astype(np.float32)),
                    "scores": torch.from_numpy(scores.astype(np.float32)),
                    "labels": torch.from_numpy(pred_i["category_id"].to_numpy().astype(np.int64)),
                }]
                targets_tensor = [{
                    "boxes": torch.from_numpy(gt_i[["x_min", "y_min", "x_max", "y_max"]].to_numpy().astype(np.float32)),
                    "labels": torch.from_numpy(gt_i["category_id"].to_numpy().astype(np.int64)),
                }]
                metrics = m_ap(preds=preds_tensor, target=targets_tensor)
                map50 = float(metrics["map_50"].item())
                map75 = float(metrics["map_75"].item())
            except Exception:
                pass

        # TP, FP, FN calculation
        if num_preds == 0 and num_gts == 0:
            tp, fp, fn = 0, 0, 0
        elif num_preds > 0 and num_gts == 0:
            tp, fp, fn = 0, num_preds, 0
        elif num_preds == 0 and num_gts > 0:
            tp, fp, fn = 0, 0, num_gts
        else:
            boxes_pred = pred_i[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
            boxes_gt = gt_i[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
            pred_cats = pred_i["category_id"].to_numpy()
            gt_cats = gt_i["category_id"].to_numpy()

            ious = compute_box_iou(boxes_pred, boxes_gt)

            # Sort predictions in descending order of confidence score
            sort_indices = np.argsort(-scores)

            matched_gts = set()
            tp_count = 0
            fp_count = 0

            for p_idx in sort_indices:
                best_gt_idx = -1
                best_iou = -1.0
                for g_idx in range(num_gts):
                    if g_idx in matched_gts:
                        continue
                    if not single_cls and pred_cats[p_idx] != gt_cats[g_idx]:
                        continue
                    iou_val = ious[p_idx, g_idx]
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = g_idx

                if best_iou >= tp_iou_threshold and best_gt_idx >= 0:
                    tp_count += 1
                    matched_gts.add(best_gt_idx)
                else:
                    fp_count += 1

            fn_count = num_gts - len(matched_gts)
            tp, fp, fn = tp_count, fp_count, fn_count

        fp_tp_ratio = fp / (tp + 1e-8)
        fn_tp_ratio = fn / (tp + 1e-8)

        per_img_records.append({
            "file_name": img_path,
            "pred_TP": tp,
            "pred_FP": fp,
            "gt_FN": fn,
            "fp_tp_ratio": fp_tp_ratio,
            "fn_tp_ratio": fn_tp_ratio,
            "map50": map50,
            "map75": map75,
            "max_score": max_score,
            "uncertainty": uncertainty,
        })

    return pd.DataFrame(per_img_records)


def select_hard_samples(
    df_results: pd.DataFrame,
    fp_tp_ratio_threshold: float = 0.2,
    fn_tp_ratio_threshold: float = 0.2,
    map_threshold: float = 0.35,
    score_threshold: float = 0.7,
    uncertainty_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Selects hard negative / challenging samples based on error ratios, low mAP, and uncertainty.
    """
    # 1. High False Positives or False Negatives
    mask_fp_fn = (
        (df_results["fp_tp_ratio"] > fp_tp_ratio_threshold) & (df_results["pred_FP"] > 0)
    ) | (
        (df_results["fn_tp_ratio"] > fn_tp_ratio_threshold) & (df_results["gt_FN"] > 0)
    )

    # 2. Low mAP with high or low confidence, or high uncertainty
    mask_low_map = (df_results["map50"] < map_threshold) & (df_results["map75"] < map_threshold)
    mask_score = (df_results["max_score"] > score_threshold) | (
        (df_results["max_score"] < (1.0 - score_threshold)) & (df_results["max_score"] > 0)
    )
    mask_uncertainty = df_results["uncertainty"] > uncertainty_threshold

    mask_map_uncertainty = (mask_low_map & mask_score) | mask_uncertainty

    mask_selected = mask_fp_fn | mask_map_uncertainty
    df_hard = df_results.loc[mask_selected].copy().drop_duplicates(subset=["file_name"]).reset_index(drop=True)

    logger.info(
        f"Hard sample filtering: {len(df_hard)} challenging samples selected "
        f"(out of {len(df_results)} evaluated)."
    )
    return df_hard


def generate_hard_negative_dataset(
    original_yaml_path: str,
    hard_sample_paths: List[str],
    save_dir: str,
) -> Tuple[str, str]:
    """
    Writes hard_samples.txt and hard_samples.yaml in save_dir.
    Returns (txt_path, yaml_path).
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    txt_path = save_dir_path / "hard_samples.txt"
    yaml_path = save_dir_path / "hard_samples.yaml"

    with open(txt_path, "w", encoding="utf-8") as f:
        for p in hard_sample_paths:
            f.write(f"{Path(p).resolve()}\n")

    orig_cfg = load_yaml(original_yaml_path)
    root = orig_cfg.get("path", "")

    # Build validation references (resolve to absolute paths)
    val_entry = orig_cfg.get("val")
    if val_entry:
        if isinstance(val_entry, list):
            val_refs = [
                str(Path(os.path.join(root, p) if root and not os.path.isabs(p) else p).resolve())
                for p in val_entry
            ]
        else:
            val_refs = str(Path(os.path.join(root, val_entry) if root and not os.path.isabs(val_entry) else val_entry).resolve())
    else:
        val_refs = []

    hard_cfg = {
        "path": str(save_dir_path.resolve()),
        "train": str(txt_path.resolve()),
        "val": val_refs,
        "names": orig_cfg.get("names", {}),
        "nc": orig_cfg.get("nc", len(orig_cfg.get("names", {}))),
    }

    save_yaml(hard_cfg, str(yaml_path))
    logger.info(f"Saved hard sample list: {txt_path}")
    logger.info(f"Saved hard sample dataset configuration: {yaml_path}")
    return str(txt_path), str(yaml_path)


# ==========================================
# 5. Training Runner
# ==========================================

def run_training(args: argparse.Namespace, data_yaml_path: str) -> str:
    """Executes YOLO fine-tuning on the hard samples dataset."""
    logger.info(f"Starting Ultralytics YOLO training using data config: {data_yaml_path}")

    model = YOLO(args.path_weights)
    run_name = f"{args.run_name}-HN_freeze_{args.hn_freeze}"

    train_args = dict(
        data=data_yaml_path,
        epochs=args.hn_num_epochs,
        imgsz=args.hn_imgsz,
        device=args.device,
        freeze=args.hn_freeze,
        name=run_name,
        single_cls=args.single_cls,
        lr0=args.hn_lr0,
        lrf=args.hn_lrf,
        momentum=args.optimizer_momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        batch=args.hn_batch_size,
        val=args.val,
        plots=True,
        cos_lr=args.cos_annealing,
        deterministic=False,
        cache=False,
        optimizer=args.optimizer,
        project=args.project_name,
        patience=args.patience,
        multi_scale=False,
        degrees=args.rotation_degree,
        mixup=args.mixup,
        scale=args.scale,
        iou=args.yolo_iou_val,
        mosaic=args.mosaic,
        augment=False,
        erasing=args.erasing,
        copy_paste=args.copy_paste,
        shear=args.shear,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        translate=args.translate,
        seed=args.seed,
        exist_ok=True,
    )

    results = model.train(**train_args)

    best_weights = getattr(results, "save_dir", None)
    if best_weights:
        best_pt = os.path.join(best_weights, "weights", "best.pt")
        logger.info(f"Training completed successfully! Best model weights saved at: {best_pt}")
        return best_pt
    else:
        logger.info("Training completed successfully!")
        return ""


# ==========================================
# 6. Main Orchestrator & CLI Arguments
# ==========================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Hard Negative Sampling and Fine-Tuning Script for YOLO models."
    )

    # Config file support
    parser.add_argument("--config", type=str, default=None, help="Path to an optional YAML configuration file.")

    # Core paths
    parser.add_argument("--path-weights", type=str, required=True, help="Path to initial model weights (.pt).")
    parser.add_argument("--hn-data-config-yaml", type=str, required=True, help="Path to dataset YAML config.")
    parser.add_argument("--hn-save-dir", type=str, required=True, help="Directory to save hard samples & artifacts.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate for hard samples.")
    parser.add_argument("--img-glob-pattern", type=str, default="*", help="File glob pattern for images.")

    # Hard sample sampling & thresholds
    parser.add_argument(
        "--hn-ratio",
        type=float,
        default=20.0,
        help="Ratio (empty/non-empty) for negative pre-sampling. Set <= 0 to evaluate on all images.",
    )
    parser.add_argument("--seed", type=int, default=41, help="Random seed.")
    parser.add_argument("--tp-iou-threshold", type=float, default=0.4, help="IoU threshold for True Positive match.")
    parser.add_argument("--fp-tp-ratio-threshold", type=float, default=0.2, help="FP/TP ratio threshold.")
    parser.add_argument("--fn-tp-ratio-threshold", type=float, default=0.2, help="FN/TP ratio threshold.")
    parser.add_argument("--hn-map-thrs", type=float, default=0.35, help="mAP threshold for hard sample identification.")
    parser.add_argument("--hn-score-thrs", type=float, default=0.7, help="Score threshold for hard sample identification.")
    parser.add_argument("--hn-uncertainty-thrs", type=float, default=5.0, help="Uncertainty threshold.")
    parser.add_argument("--hn-uncertainty-method", type=str, default="entropy", choices=["entropy", "1-p"])

    # Inference settings
    parser.add_argument("--use-sliding-window", action="store_true", default=True, help="Use SAHI sliced inference.")
    parser.add_argument("--no-sliding-window", action="store_false", dest="use_sliding_window")
    parser.add_argument("--hn-tilesize", type=int, default=1280, help="Tile size for SAHI sliced inference.")
    parser.add_argument("--hn-overlap-ratio", type=float, default=0.2, help="Tile overlap ratio for SAHI.")
    parser.add_argument("--hn-confidence-threshold", type=float, default=0.25, help="Confidence threshold for inference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Training settings
    parser.add_argument("--hn-num-epochs", type=int, default=10, help="Number of training epochs on hard samples.")
    parser.add_argument("--hn-imgsz", type=int, default=1280, help="Image size for training and inference.")
    parser.add_argument("--hn-batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--hn-freeze", type=int, default=20, help="Number of layers to freeze in YOLO.")
    parser.add_argument("--hn-lr0", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--hn-lrf", type=float, default=1e-1, help="Final learning rate factor.")
    parser.add_argument("--single-cls", action="store_true", default=False, help="Train and evaluate as single-class dataset.")
    parser.add_argument("--val", action="store_true", default=True, help="Run validation during training.")
    parser.add_argument("--no-val", action="store_false", dest="val", help="Disable validation during training.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer name.")
    parser.add_argument("--optimizer-momentum", type=float, default=0.99, help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--cos-annealing", action="store_true", default=True, help="Cosine LR scheduling.")
    parser.add_argument("--no-cos-annealing", action="store_false", dest="cos_annealing", help="Disable cosine annealing LR.")
    parser.add_argument("--yolo-iou-val", type=float, default=0.6, help="NMS IoU threshold for validation.")

    # Project and run name
    parser.add_argument("--run-name", type=str, default="wildAI-HN", help="Run name.")
    parser.add_argument("--project-name", type=str, default="wildAI", help="Project name.")

    # Data augmentations
    parser.add_argument("--rotation-degree", type=float, default=45.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--shear", type=float, default=10.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--erasing", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.3)
    parser.add_argument("--hsv-v", type=float, default=0.3)
    parser.add_argument("--translate", type=float, default=0.2)
    parser.add_argument("--mosaic", type=float, default=0.0)

    # Execution control
    parser.add_argument("--skip-train", action="store_true", default=False, help="Only sample hard negatives without training.")

    args = parser.parse_args()

    # If --config is supplied, override defaults from YAML
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configurations from {args.config}")
        cfg_dict = load_yaml(args.config)
        for key, val in cfg_dict.items():
            attr_name = key.replace("-", "_")
            if hasattr(args, attr_name):
                setattr(args, attr_name, val)

    return args


def main():
    args = parse_args()
    os.makedirs(args.hn_save_dir, exist_ok=True)

    logger.info("=================================================================")
    logger.info("        Hard Negative Sampling and Learning Pipeline             ")
    logger.info("=================================================================")

    # Step 1: Remove label cache
    logger.info("\n[1/5] Removing stale dataset labels cache...")
    remove_label_cache(args.hn_data_config_yaml)

    # Step 2: Discover and optionally pre-sample candidate images
    logger.info("\n[2/5] Discovering images for hard negative evaluation...")
    data_cfg = load_yaml(args.hn_data_config_yaml)
    candidate_images = get_images_from_split(
        yolo_config=data_cfg,
        split=args.split,
        pattern_glob=args.img_glob_pattern,
    )
    logger.info(f"Found {len(candidate_images)} total images in split '{args.split}'.")

    if args.hn_ratio > 0:
        logger.info(f"Pre-sampling negative images with empty ratio={args.hn_ratio}...")
        candidate_images = sample_pos_neg(
            images_paths=candidate_images,
            ratio=args.hn_ratio,
            seed=args.seed,
        )
        logger.info(f"{len(candidate_images)} candidate images selected after pre-sampling.")

    if len(candidate_images) == 0:
        logger.error("No candidate images available for evaluation. Exiting.")
        sys.exit(1)

    # Step 3: Run inference and load ground truth
    logger.info("\n[3/5] Running detector inference & loading ground truth...")
    detector = DetectorInference(
        weights_path=args.path_weights,
        device=args.device,
        use_sliding_window=args.use_sliding_window,
        confidence_threshold=args.hn_confidence_threshold,
        overlap_ratio=args.hn_overlap_ratio,
        tilesize=args.hn_tilesize,
        imgsz=args.hn_imgsz,
    )
    df_pred = detector.predict_images(candidate_images)
    df_gt, num_empty = load_yolo_groundtruth(candidate_images)

    logger.info(f"Detections produced: {len(df_pred)} boxes across {df_pred['file_name'].nunique() if not df_pred.empty else 0} images.")
    logger.info(f"Ground truth loaded: {len(df_gt)} boxes ({num_empty} empty images).")

    # Step 4: Evaluate performance & select hard samples
    logger.info("\n[4/5] Evaluating detection metrics and selecting hard samples...")
    df_metrics = evaluate_image_metrics(
        candidate_images=candidate_images,
        df_pred=df_pred,
        df_gt=df_gt,
        tp_iou_threshold=args.tp_iou_threshold,
        uncertainty_method=args.hn_uncertainty_method,
        single_cls=args.single_cls,
    )

    metrics_csv_path = os.path.join(args.hn_save_dir, "evaluation_metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"Per-image evaluation metrics saved to {metrics_csv_path}")

    df_hard = select_hard_samples(
        df_results=df_metrics,
        fp_tp_ratio_threshold=args.fp_tp_ratio_threshold,
        fn_tp_ratio_threshold=args.fn_tp_ratio_threshold,
        map_threshold=args.hn_map_thrs,
        score_threshold=args.hn_score_thrs,
        uncertainty_threshold=args.hn_uncertainty_thrs,
    )

    hard_samples_csv_path = os.path.join(args.hn_save_dir, "hard_samples.csv")
    df_hard.to_csv(hard_samples_csv_path, index=False)

    hard_image_paths = df_hard["file_name"].tolist()
    if len(hard_image_paths) == 0:
        logger.warning("No hard samples were flagged with the current thresholds.")
        if not args.skip_train:
            logger.info("Proceeding with full candidate images set for training.")
            hard_image_paths = candidate_images

    txt_path, yaml_path = generate_hard_negative_dataset(
        original_yaml_path=args.hn_data_config_yaml,
        hard_sample_paths=hard_image_paths,
        save_dir=args.hn_save_dir,
    )

    # Step 5: Fine-tune model on hard samples
    if args.skip_train:
        logger.info("\n[5/5] --skip-train provided. Skipping fine-tuning.")
    else:
        logger.info("\n[5/5] Starting fine-tuning on hard samples dataset...")
        best_pt = run_training(args, data_yaml_path=yaml_path)
        logger.info(f"\nPipeline finished! Best checkpoint: {best_pt}")


if __name__ == "__main__":
    main()
