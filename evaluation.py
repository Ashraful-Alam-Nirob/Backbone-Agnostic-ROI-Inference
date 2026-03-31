import os
import time
import copy
import json
import random
import warnings
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# CONFIG — EDIT THESE
# =========================

# Your split folder must look like:
# ROOT_SPLIT_DIR/
#   train/T1/<class>/*.png
#   val/T1/<class>/*.png
#   test/T1/<class>/*.png
#   train/T2/...
#   val/T2/...
#   test/T2/...
#   train/T1C+/...
ROOT_SPLIT_DIR = r"/home/nirob/Desktop/Brain tumor mri /dataset/data_split"

MODALITIES = ["T1", "T2", "T1C+"]  # edit if your folder name differs

# Where to save everything
OUTPUT_DIR = r"/home/nirob/Desktop/Brain tumor mri /benchmark_outputs_timm_q1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# MODELS (put your 12 here)
# -------------------------
MODEL_IDS = {
    "Inception-V3": "inception_v3.gluon_in1k",
    "MobileNetV3-L": "mobilenetv3_large_100.miil_in21k",
    "VGG-19-BN": "vgg19_bn.tv_in1k",
    "DenseNet-121": "densenet121.ra_in1k",
    "Xception41": "xception41.tf_in1k",
    "VGG-16-BN": "vgg16_bn.tv_in1k",
    "CoatNet-0": "coatnet_0_rw_224.sw_in1k",
    "EfficientNet-B0": "efficientnet_b0.ra4_e3600_r224_in1k",
    "MaxViT-Tiny-256": "maxvit_rmlp_tiny_rw_256.sw_in1k",
    "ConvNeXtV2-Tiny": "convnextv2_tiny.fcmae",
    "ConvNeXtV2-Atto": "convnextv2_atto.fcmae",
    "Swin-T": "swin_tiny_patch4_window7_224.ms_in22k"
}

# -------------------------
# TRAIN SETTINGS
# -------------------------
EPOCHS = 50
LR = 5e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0
BATCH_SIZE = 16
NUM_WORKERS = 4
GRAD_CLIP = 1.0

# Mixed precision
USE_AMP_TRAIN = True
USE_AMP_EVAL = True
USE_AMP_INFER_BENCH = True

# Benchmarking
WARMUP_BATCHES = 10
BENCHMARK_MAX_BATCHES = None  # e.g. 30 to shorten

# Seed (single seed now, easy to extend later)
SEEDS = [24,48]

# Splits
EVAL_TEST = False  # requires ROOT_SPLIT_DIR/test/<MODALITY> to exist

# Fair speed/latency comparisons:
# True => forces all models to same input size
FORCE_FIXED_IMAGE_SIZE = True
FIXED_IMAGE_SIZE = 192  # your proposed default

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# REPRO + PERF
# =========================
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


# =========================
# METRICS
# =========================
def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_p": float(p),
        "macro_r": float(r),
        "macro_f1": float(f1),
        "w_f1": float(f1w),
    }

def macro_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Optional[float]:
    # Can fail if some classes absent in a split; handle safely.
    try:
        y_true_oh = np.eye(num_classes)[y_true]
        auc = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
        return float(auc)
    except Exception:
        return None

def mean_max_confidence(y_prob: np.ndarray) -> float:
    return float(np.mean(np.max(y_prob, axis=1)))

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)

        if np.any(mask):
            bin_acc = float(np.mean(acc[mask]))
            bin_conf = float(np.mean(conf[mask]))
            ece += (np.sum(mask) / len(conf)) * abs(bin_acc - bin_conf)
    return float(ece)

def nll_from_probs(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = y_prob[np.arange(len(y_true)), y_true]
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.log(p)))

def brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    y_oh = np.eye(num_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((y_prob - y_oh) ** 2, axis=1)))

def params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    num_classes = len(class_names)
    met = basic_metrics(y_true, y_pred)
    auc = macro_auc_ovr(y_true, y_prob, num_classes)
    conf = mean_max_confidence(y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    nll = nll_from_probs(y_true, y_prob)
    brier = brier_multiclass(y_true, y_prob, num_classes)
    return {
        **met,
        "auc_macro_ovr": float(auc) if auc is not None else np.nan,
        "mean_confidence": conf,
        "ece_15bins": ece,
        "nll": nll,
        "brier": brier,
    }


# =========================
# ARTIFACT HELPERS
# =========================
def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def calibration_bins_df(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)

        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = float(np.mean(acc[mask]))
            bin_conf = float(np.mean(conf[mask]))
            gap = float(abs(bin_acc - bin_conf))
        else:
            bin_acc, bin_conf, gap = np.nan, np.nan, np.nan

        rows.append({
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "count": cnt,
            "bin_acc": bin_acc,
            "bin_conf": bin_conf,
            "gap": gap,
        })
    return pd.DataFrame(rows)

def risk_coverage_df(y_true: np.ndarray, y_prob: np.ndarray, num_points: int = 101) -> pd.DataFrame:
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    err = (pred != y_true).astype(np.float32)

    order = np.argsort(-conf)  # high conf first
    err_sorted = err[order]
    conf_sorted = conf[order]

    n = len(y_true)
    rows = []
    for k in np.linspace(0.01, 1.0, num_points):
        m = max(1, int(round(k * n)))
        risk = float(np.mean(err_sorted[:m]))
        thr = float(conf_sorted[m - 1])
        rows.append({"coverage": float(m / n), "risk": risk, "threshold": thr})
    return pd.DataFrame(rows)

def save_eval_arrays_npz(out_path: str, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]):
    np.savez_compressed(
        out_path,
        y_true=y_true.astype(np.int64),
        y_prob=y_prob.astype(np.float32),
        class_names=np.array(class_names),
    )

def save_roc_pr_npz(out_path: str, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]):
    num_classes = len(class_names)
    y_true_oh = np.eye(num_classes, dtype=np.int32)[y_true]

    data = {}
    for c in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_prob[:, c])
            data[f"roc_fpr_{c}"] = fpr.astype(np.float32)
            data[f"roc_tpr_{c}"] = tpr.astype(np.float32)
        except Exception:
            data[f"roc_fpr_{c}"] = np.array([], dtype=np.float32)
            data[f"roc_tpr_{c}"] = np.array([], dtype=np.float32)

        try:
            prec, rec, _ = precision_recall_curve(y_true_oh[:, c], y_prob[:, c])
            data[f"pr_prec_{c}"] = prec.astype(np.float32)
            data[f"pr_rec_{c}"] = rec.astype(np.float32)
        except Exception:
            data[f"pr_prec_{c}"] = np.array([], dtype=np.float32)
            data[f"pr_rec_{c}"] = np.array([], dtype=np.float32)

    data["class_names"] = np.array(class_names)
    np.savez_compressed(out_path, **data)

def sanitize_name(s: str) -> str:
    return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])

def save_confusion_outputs(cm: np.ndarray, class_names: List[str], out_prefix: str, title: str):
    # Raw CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = out_prefix + "_confusion_matrix.csv"
    cm_df.to_csv(cm_csv, index=True)

    # Normalized CSV (row-normalized)
    cm_norm = cm.astype(np.float32)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, np.maximum(row_sum, 1e-12))
    cmn_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cmn_csv = out_prefix + "_confusion_matrix_norm.csv"
    cmn_df.to_csv(cmn_csv, index=True)

    # Heatmap PNG (raw)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=7,
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    cm_png = out_prefix + "_confusion_matrix.png"
    fig.savefig(cm_png, dpi=200)
    plt.close(fig)

    return cm_csv, cmn_csv, cm_png


# =========================
# DATA LOADERS (per model)
# =========================
def make_loaders_for_model(
    model: torch.nn.Module,
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None
):
    cfg = resolve_data_config({}, model=model)

    # Force same input size for fair latency/throughput comparisons
    if FORCE_FIXED_IMAGE_SIZE:
        cfg["input_size"] = (3, FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE)

    train_tfms = create_transform(**cfg, is_training=True)
    eval_tfms = create_transform(**cfg, is_training=False)

    train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_set   = datasets.ImageFolder(val_dir, transform=eval_tfms)

    # Ensure class mapping consistent
    if train_set.classes != val_set.classes:
        raise ValueError(f"Class mismatch train vs val: {train_set.classes} vs {val_set.classes}")

    test_set = None
    test_loader = None
    if test_dir is not None and os.path.isdir(test_dir):
        test_set = datasets.ImageFolder(test_dir, transform=eval_tfms)
        if train_set.classes != test_set.classes:
            raise ValueError(f"Class mismatch train vs test: {train_set.classes} vs {test_set.classes}")
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, cfg


# =========================
# EVAL + BENCH
# =========================
@torch.no_grad()
def eval_model(model: torch.nn.Module, loader: DataLoader, amp: bool):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(amp and DEVICE == "cuda")):
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            prob = torch.softmax(out, dim=1)
        pred = prob.argmax(dim=1)

        y_true.append(yb.numpy())
        y_pred.append(pred.cpu().numpy())
        y_prob.append(prob.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

@torch.no_grad()
def benchmark_infer(
    model: torch.nn.Module,
    loader: DataLoader,
    amp: bool,
    warmup_batches: int = 10,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    model.eval()

    # warmup
    w = 0
    for xb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(amp and DEVICE == "cuda")):
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
        w += 1
        if w >= warmup_batches:
            break

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    total_s = 0.0
    total_imgs = 0
    total_batches = 0

    for xb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=(amp and DEVICE == "cuda")):
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_s += (t1 - t0)
        total_imgs += int(xb.size(0))
        total_batches += 1

        if max_batches is not None and total_batches >= max_batches:
            break

    ms_per_img = (total_s * 1000.0) / max(total_imgs, 1)
    throughput = total_imgs / max(total_s, 1e-12)
    return float(ms_per_img), float(throughput)


# =========================
# TRAIN
# =========================
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp: bool,
    label_smoothing: float
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and DEVICE == "cuda")):
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = F.cross_entropy(out, yb, label_smoothing=label_smoothing)

        if scaler is not None and (DEVICE == "cuda") and amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


# =========================
# RUN ONE MODEL (one modality, one seed)
# =========================
def run_one(
    modality: str,
    model_name: str,
    timm_id: str,
    seed: int,
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str]
) -> Dict[str, object]:
    safe_model = sanitize_name(model_name)
    run_tag = f"{safe_model}_seed{seed}"
    model_out_dir = os.path.join(OUTPUT_DIR, modality, safe_model, f"seed_{seed}")
    os.makedirs(model_out_dir, exist_ok=True)

    epoch_csv_path = os.path.join(model_out_dir, f"{run_tag}_epoch_log.csv")
    final_csv_path = os.path.join(model_out_dir, f"{run_tag}_final_metrics.csv")

    print("\n" + "=" * 80)
    print(f"[MODALITY] {modality} | [MODEL] {model_name} | timm id: {timm_id} | seed: {seed}")
    print("=" * 80)
    print("Outputs:", model_out_dir)

    # Create model
    model = timm.create_model(timm_id, pretrained=True, num_classes=0)  # create first to resolve cfg
    model = model.to(DEVICE)

    # Build loaders using model cfg (then re-create model with correct head)
    train_set, val_set, test_set, train_loader, val_loader, test_loader, cfg = make_loaders_for_model(
        model, train_dir, val_dir, test_dir
    )
    class_names = train_set.classes
    num_classes = len(class_names)

    # Re-create with correct head (some models need num_classes at init)
    model = timm.create_model(timm_id, pretrained=True, num_classes=num_classes).to(DEVICE)

    meta_path = os.path.join(model_out_dir, f"{run_tag}_meta.json")
    save_json(meta_path, {
        "modality": modality,
        "model_name": model_name,
        "timm_id": timm_id,
        "seed": seed,
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "use_amp_train": USE_AMP_TRAIN,
        "use_amp_eval": USE_AMP_EVAL,
        "use_amp_infer_bench": USE_AMP_INFER_BENCH,
        "force_fixed_image_size": FORCE_FIXED_IMAGE_SIZE,
        "fixed_image_size": FIXED_IMAGE_SIZE if FORCE_FIXED_IMAGE_SIZE else None,
        "resolved_cfg": {k: (v if isinstance(v, (int, float, str, bool, list, dict, tuple)) else str(v)) for k, v in cfg.items()},
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir if (test_loader is not None) else None,
        "class_names": class_names,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP_TRAIN and DEVICE == "cuda"))

    best_score = -1.0
    best_state = None
    best_epoch = -1

    epoch_rows = []

    for ep in range(1, EPOCHS + 1):
        t0 = time.perf_counter()

        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, USE_AMP_TRAIN, LABEL_SMOOTHING)
        y_true, y_pred, y_prob = eval_model(model, val_loader, amp=USE_AMP_EVAL)

        val_met = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        t1 = time.perf_counter()

        print(
            f"Epoch {ep:02d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} | val_acc={val_met['acc']:.4f} | "
            f"val_macro_f1={val_met['macro_f1']:.4f} | val_ece={val_met['ece_15bins']:.4f}"
        )

        epoch_rows.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_acc": val_met["acc"],
            "val_macro_p": val_met["macro_p"],
            "val_macro_r": val_met["macro_r"],
            "val_macro_f1": val_met["macro_f1"],
            "val_weighted_f1": val_met["w_f1"],
            "val_auc_macro_ovr": val_met["auc_macro_ovr"],
            "val_mean_confidence": val_met["mean_confidence"],
            "val_ece_15bins": val_met["ece_15bins"],
            "val_nll": val_met["nll"],
            "val_brier": val_met["brier"],
            "epoch_time_s": float(t1 - t0),
        })

        # Save epoch CSV every epoch
        pd.DataFrame(epoch_rows).to_csv(epoch_csv_path, index=False)

        # Best checkpoint by val_acc (as in your current protocol)
        score = val_met["acc"]
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())

            best_ckpt_path = os.path.join(model_out_dir, f"{run_tag}_best.pth")
            torch.save({
                "modality": modality,
                "model_name": model_name,
                "timm_id": timm_id,
                "seed": seed,
                "num_classes": num_classes,
                "class_names": class_names,
                "best_acc": float(best_score),
                "best_epoch": int(best_epoch),
                "state_dict": best_state,
            }, best_ckpt_path)

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # -----------------------
    # FINAL EVAL: VAL
    # -----------------------
    y_true_v, y_pred_v, y_prob_v = eval_model(model, val_loader, amp=USE_AMP_EVAL)
    val_met = compute_all_metrics(y_true_v, y_pred_v, y_prob_v, class_names)

    # Confusion outputs (VAL)
    cm_v = confusion_matrix(y_true_v, y_pred_v)
    prefix_v = os.path.join(model_out_dir, f"{run_tag}_VAL")
    val_cm_csv, val_cmn_csv, val_cm_png = save_confusion_outputs(cm_v, class_names, prefix_v, title="VAL Confusion Matrix")

    # Calibration artifacts (VAL)
    val_arrays_npz = os.path.join(model_out_dir, f"{run_tag}_VAL_arrays.npz")
    save_eval_arrays_npz(val_arrays_npz, y_true_v, y_prob_v, class_names)

    val_calib_bins_csv = os.path.join(model_out_dir, f"{run_tag}_VAL_calibration_bins.csv")
    calibration_bins_df(y_true_v, y_prob_v, n_bins=15).to_csv(val_calib_bins_csv, index=False)

    val_riskcov_csv = os.path.join(model_out_dir, f"{run_tag}_VAL_risk_coverage.csv")
    risk_coverage_df(y_true_v, y_prob_v).to_csv(val_riskcov_csv, index=False)

    val_rocpr_npz = os.path.join(model_out_dir, f"{run_tag}_VAL_roc_pr_curves.npz")
    save_roc_pr_npz(val_rocpr_npz, y_true_v, y_prob_v, class_names)

    # -----------------------
    # FINAL EVAL: TEST (optional)
    # -----------------------
    test_met = None
    test_outputs = {}
    if test_loader is not None:
        y_true_t, y_pred_t, y_prob_t = eval_model(model, test_loader, amp=USE_AMP_EVAL)
        test_met = compute_all_metrics(y_true_t, y_pred_t, y_prob_t, class_names)

        cm_t = confusion_matrix(y_true_t, y_pred_t)
        prefix_t = os.path.join(model_out_dir, f"{run_tag}_TEST")
        test_cm_csv, test_cmn_csv, test_cm_png = save_confusion_outputs(cm_t, class_names, prefix_t, title="TEST Confusion Matrix")

        test_arrays_npz = os.path.join(model_out_dir, f"{run_tag}_TEST_arrays.npz")
        save_eval_arrays_npz(test_arrays_npz, y_true_t, y_prob_t, class_names)

        test_calib_bins_csv = os.path.join(model_out_dir, f"{run_tag}_TEST_calibration_bins.csv")
        calibration_bins_df(y_true_t, y_prob_t, n_bins=15).to_csv(test_calib_bins_csv, index=False)

        test_riskcov_csv = os.path.join(model_out_dir, f"{run_tag}_TEST_risk_coverage.csv")
        risk_coverage_df(y_true_t, y_prob_t).to_csv(test_riskcov_csv, index=False)

        test_rocpr_npz = os.path.join(model_out_dir, f"{run_tag}_TEST_roc_pr_curves.npz")
        save_roc_pr_npz(test_rocpr_npz, y_true_t, y_prob_t, class_names)

        test_outputs = {
            "test_confusion_csv": test_cm_csv,
            "test_confusion_norm_csv": test_cmn_csv,
            "test_confusion_png": test_cm_png,
            "test_arrays_npz": test_arrays_npz,
            "test_calibration_bins_csv": test_calib_bins_csv,
            "test_risk_coverage_csv": test_riskcov_csv,
            "test_rocpr_npz": test_rocpr_npz,
        }

    # -----------------------
    # EFFICIENCY BENCH (use VAL loader)
    # -----------------------
    ms_img, thr = benchmark_infer(
        model, val_loader,
        amp=USE_AMP_INFER_BENCH,
        warmup_batches=WARMUP_BATCHES,
        max_batches=BENCHMARK_MAX_BATCHES
    )

    final_row = {
        "modality": modality,
        "model": model_name,
        "timm_id": timm_id,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_by": "val_acc",

        # VAL metrics
        "val_acc": val_met["acc"],
        "val_macro_p": val_met["macro_p"],
        "val_macro_r": val_met["macro_r"],
        "val_macro_f1": val_met["macro_f1"],
        "val_w_f1": val_met["w_f1"],
        "val_auc_macro_ovr": val_met["auc_macro_ovr"],
        "val_mean_confidence": val_met["mean_confidence"],
        "val_ece_15bins": val_met["ece_15bins"],
        "val_nll": val_met["nll"],
        "val_brier": val_met["brier"],

        # TEST metrics (if available)
        "test_acc": (test_met["acc"] if test_met is not None else np.nan),
        "test_macro_f1": (test_met["macro_f1"] if test_met is not None else np.nan),
        "test_auc_macro_ovr": (test_met["auc_macro_ovr"] if test_met is not None else np.nan),
        "test_ece_15bins": (test_met["ece_15bins"] if test_met is not None else np.nan),
        "test_nll": (test_met["nll"] if test_met is not None else np.nan),
        "test_brier": (test_met["brier"] if test_met is not None else np.nan),

        # efficiency
        "params_m": params_m(model),
        "ms_per_img": ms_img,
        "throughput_img_s": thr,

        # protocol flags
        "force_fixed_image_size": FORCE_FIXED_IMAGE_SIZE,
        "fixed_image_size": (FIXED_IMAGE_SIZE if FORCE_FIXED_IMAGE_SIZE else np.nan),
        "use_amp_train": USE_AMP_TRAIN,
        "use_amp_eval": USE_AMP_EVAL,
        "use_amp_infer_bench": USE_AMP_INFER_BENCH,

        # artifact pointers
        "meta_json": meta_path,
        "epoch_log_csv": epoch_csv_path,

        "val_confusion_csv": val_cm_csv,
        "val_confusion_norm_csv": val_cmn_csv,
        "val_confusion_png": val_cm_png,
        "val_arrays_npz": val_arrays_npz,
        "val_calibration_bins_csv": val_calib_bins_csv,
        "val_risk_coverage_csv": val_riskcov_csv,
        "val_rocpr_npz": val_rocpr_npz,
        **test_outputs,
    }

    pd.DataFrame([final_row]).to_csv(final_csv_path, index=False)

    print("\n[FINAL (VAL)]")
    print(f"Acc={final_row['val_acc']:.4f} | MacroF1={final_row['val_macro_f1']:.4f} | "
          f"AUC={final_row['val_auc_macro_ovr']:.4f} | ECE={final_row['val_ece_15bins']:.4f} | "
          f"NLL={final_row['val_nll']:.4f} | Brier={final_row['val_brier']:.4f}")

    if test_met is not None:
        print("[FINAL (TEST)]")
        print(f"Acc={final_row['test_acc']:.4f} | MacroF1={final_row['test_macro_f1']:.4f} | "
              f"AUC={final_row['test_auc_macro_ovr']:.4f} | ECE={final_row['test_ece_15bins']:.4f} | "
              f"NLL={final_row['test_nll']:.4f} | Brier={final_row['test_brier']:.4f}")

    print("\n[EFFICIENCY]")
    print(f"Params(M)={final_row['params_m']:.2f} | ms/img={final_row['ms_per_img']:.4f} | img/s={final_row['throughput_img_s']:.2f}")

    print("\n[SAVED]")
    print("Final metrics CSV:", final_csv_path)
    print("Epoch log CSV    :", epoch_csv_path)
    return final_row


# =========================
# MAIN (loop modalities + models + seeds)
# =========================
def main():
    print("Device:", DEVICE)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("FORCE_FIXED_IMAGE_SIZE:", FORCE_FIXED_IMAGE_SIZE, "| FIXED_IMAGE_SIZE:", FIXED_IMAGE_SIZE)

    all_results = []
    all_fail = []

    for modality in MODALITIES:
        train_dir = os.path.join(ROOT_SPLIT_DIR, "train", modality)
        val_dir   = os.path.join(ROOT_SPLIT_DIR, "val", modality)
        test_dir  = os.path.join(ROOT_SPLIT_DIR, "test", modality) if EVAL_TEST else None

        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            print(f"\n[SKIP] Missing dirs for modality {modality}:")
            print(" train_dir:", train_dir)
            print(" val_dir  :", val_dir)
            continue

        print("\n" + "#" * 90)
        print(f"MODALITY: {modality}")
        print("#" * 90)
        print("train_dir:", train_dir)
        print("val_dir  :", val_dir)
        print("test_dir :", test_dir if (test_dir and os.path.isdir(test_dir)) else "None/Not found")

        for seed in SEEDS:
            set_seed(seed)

            for name, mid in MODEL_IDS.items():
                try:
                    r = run_one(
                        modality=modality,
                        model_name=name,
                        timm_id=mid,
                        seed=seed,
                        train_dir=train_dir,
                        val_dir=val_dir,
                        test_dir=(test_dir if (test_dir and os.path.isdir(test_dir)) else None),
                    )
                    all_results.append(r)
                except Exception as e:
                    err = str(e)
                    print(f"\n[ERROR] {modality} | {name} failed: {err}")
                    all_fail.append({"modality": modality, "model": name, "timm_id": mid, "seed": seed, "error": err})

        # Per-modality summary
        modality_csv = os.path.join(OUTPUT_DIR, f"SUMMARY_{sanitize_name(modality)}.csv")
        if all_results:
            dfm = pd.DataFrame([r for r in all_results if r["modality"] == modality])
            if not dfm.empty:
                # Sort by TEST if available else VAL
                sort_col = "test_acc" if dfm["test_acc"].notna().any() else "val_acc"
                dfm = dfm.sort_values(sort_col, ascending=False)
                dfm.to_csv(modality_csv, index=False)
                print("\nSaved modality summary CSV:", modality_csv)

    # Global summaries
    if all_results:
        df = pd.DataFrame(all_results)
        global_csv = os.path.join(OUTPUT_DIR, "ALL_RESULTS_GLOBAL.csv")
        df.to_csv(global_csv, index=False)
        print("\nSaved global CSV:", global_csv)

        # Convenience summary: best per modality/model (if multiple seeds later)
        best_csv = os.path.join(OUTPUT_DIR, "BEST_PER_MODEL_MODALITY.csv")
        metric = "test_acc" if df["test_acc"].notna().any() else "val_acc"
        df_best = (
            df.sort_values(metric, ascending=False)
              .groupby(["modality", "model"], as_index=False)
              .head(1)
        )
        df_best.to_csv(best_csv, index=False)
        print("Saved best-per-model CSV:", best_csv)

    if all_fail:
        fail_csv = os.path.join(OUTPUT_DIR, "FAILED_RUNS.csv")
        pd.DataFrame(all_fail).to_csv(fail_csv, index=False)
        print("\nSaved failures CSV:", fail_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()