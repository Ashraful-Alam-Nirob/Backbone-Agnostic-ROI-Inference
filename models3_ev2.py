import os
import time
import random
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# >>> NEW IMPORTS (CSV + metrics) <<<
import csv
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score, log_loss

import timm

# -----------------------
# CONFIG
# -----------------------
seed = 0
batch_size = 16
batch_size_inference = 16   
image_size = 192
patch_size = 112
roi_predictor_size = 112

K_rois = 4

epochs = 120
stage1_epochs = 20

# Tuned params
lr_stage1 = 0.0004861891514312961
lr_stage2 = 0.0001859698083735276

s_min, s_max = 0.35, 1.0

target_K = 0.9326982778020024
budget_weight = 0.10529834453051307
gate_temperature = 1.6701770112272525

# --- Inference knobs ---
tau_infer = 0.7677773160959454
Kmax_infer = 0             # FASTEST in your tests (global-only)
dynamic_kmax = False       

# Acc knobs
label_smoothing = 0.000670836038508418
mixup_alpha = 0.15184374953485258
ema_decay = 0.977319035633441
warmup_epochs = 3
use_tta = False

BACKBONE_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
clip_norm = 1.0
eps = 1e-6

# AMP for pruned inference
use_amp_pruned_infer = True

# Inference memory format
INFER_CHANNELS_LAST = True


ece_weight = 0.02          # small, safe default (try 0.01~0.05)
ece_bins = 15              # typical ECE bin count
ece_smoothing = 0.02       # soft-binning smoothness (larger => smoother)
ece_temperature = 1.0      # for the Soft-ECE loss only (not temp scaling)
SAVE_CSV  = True

# Paths
dataset = "T1"
train_dir = r"/home/nirob/Desktop/Brain tumor mri /dataset/data_split/train/T1"
val_dir   = r"/home/nirob/Desktop/Brain tumor mri /dataset/data_split/val/T1"
CKPT_SAVE_PATH = r"/home/nirob/Desktop/Brain tumor mri /checkpoints/t1_final_best_optimized_model.pth"

# >>> NEW: metrics CSV path <<<
METRICS_CSV_PATH = str(Path(CKPT_SAVE_PATH).with_name(f"{dataset}_runs_metrics.csv"))

# Performance toggles
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------
# REPRODUCIBILITY
# -----------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# DATA AUGMENTATION
# -----------------------
train_tfms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
val_set   = datasets.ImageFolder(val_dir,   transform=val_tfms)

class_names = train_set.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("num_classes:", num_classes)

# -----------------------
# CLASS-WEIGHTED SAMPLING
# -----------------------
def get_class_weights_and_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    weights_tensor = torch.FloatTensor(class_weights)
    return sampler, weights_tensor

def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps)
    return -(p * p.log()).sum(dim=1)

def margin_from_probs(p: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(p, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]

train_sampler, class_weights = get_class_weights_and_sampler(train_set)
print(f"Class weights computed: {class_weights.numpy()}")

train_loader = DataLoader(
    train_set, batch_size=batch_size, sampler=train_sampler,
    num_workers=4, pin_memory=True, drop_last=False,
    persistent_workers=True
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True, drop_last=False,
    persistent_workers=True
)
val_loader_fast = DataLoader(
    val_set, batch_size=batch_size_inference, shuffle=False,
    num_workers=4, pin_memory=True, drop_last=False,
    persistent_workers=True
)

# -----------------------
# NORMALIZATION
# -----------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def normalize_torch(x_01: torch.Tensor) -> torch.Tensor:
    return (x_01 - IMAGENET_MEAN) / IMAGENET_STD

# -----------------------
# MIXUP
# -----------------------
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------
# CONCRETE SIGMOID
# -----------------------
def concrete_sigmoid(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    u = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
    g = torch.log(u) - torch.log(1.0 - u)
    return torch.sigmoid((logits + g) / temperature)

# -----------------------
# NEW: DIFFERENTIABLE "SOFT-ECE" LOSS (calibration-aware constraint)
# -----------------------
def soft_ece_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
    smoothing: float = 0.02,
    temperature: float = 1.0,
    eps_: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits / max(temperature, 1e-6), dim=1)
    conf, _ = probs.max(dim=1)  # [B]
    ptrue = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # [B]

    B = conf.size(0)
    if B == 0:
        return logits.sum() * 0.0

    dtype = conf.dtype
    dev = conf.device

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=dev, dtype=dtype)
    lower = edges[:-1].view(1, -1)  # [1, n_bins]
    upper = edges[1:].view(1, -1)   # [1, n_bins]

    conf_col = conf.view(-1, 1)     # [B, 1]
    ptrue_col = ptrue.view(-1, 1)   # [B, 1]

    s = max(float(smoothing), 1e-6)
    w = torch.sigmoid((conf_col - lower) / s) - torch.sigmoid((conf_col - upper) / s)  # [B, n_bins]
    w = w.clamp(min=0.0)

    bin_w = w.sum(dim=0) + eps_          # [n_bins]
    total_w = bin_w.sum() + eps_

    avg_conf = (w * conf_col).sum(dim=0) / bin_w
    avg_ptrue = (w * ptrue_col).sum(dim=0) / bin_w

    diff = avg_conf - avg_ptrue
    loss = (bin_w / total_w) * (diff * diff)
    return loss.sum()

# -----------------------
# NEW: ECE from probs (numpy) + Temperature Scaling (post-hoc)
# -----------------------
def ece_from_probs(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    if probs.size == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        m = (bin_ids == b)
        if not np.any(m):
            continue
        acc_b = float(correct[m].mean())
        conf_b = float(conf[m].mean())
        ece += (m.sum() / max(N, 1)) * abs(acc_b - conf_b)
    return float(ece)

def softmax_np(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return (ez / (ez.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)

@torch.inference_mode()
def collect_val_logits(
    model,
    loader,
    cfg,
    pruned: bool,
    graph_runner=None,
    max_steps=None,
):
    model.eval()
    y_true_list = []
    logits_list = []
    steps = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        if pruned:
            if graph_runner is not None:
                logits = graph_runner(xb)
            else:
                logits = hard_pruned_predict_batch_fast(
                    model, xb,
                    tau=cfg["tau_infer"],
                    Kmax=cfg["Kmax_infer"],
                    use_dynamic=cfg.get("dynamic_kmax", False),
                    print_escalation=False,
                    entropy_thr=cfg.get("entropy_thr", None),
                    margin_thr=cfg.get("margin_thr", 0.15),
                    per_sample_escalation=cfg.get("per_sample_escalation", True),
                )
        else:
            logits = model(
                xb, training=False,
                gate_temperature=cfg["gate_temperature"],
                target_K=cfg["target_K"],
                budget_weight=cfg["budget_weight"],
            )[0]

        logits_list.append(logits.detach().float().cpu().numpy())
        y_true_list.append(yb.numpy())

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    y_true = np.concatenate(y_true_list) if len(y_true_list) else np.array([], dtype=np.int64)
    logits = np.concatenate(logits_list) if len(logits_list) else np.zeros((0, num_classes), dtype=np.float32)
    return y_true, logits

def fit_temperature_on_val(
    logits_np: np.ndarray,
    y_true_np: np.ndarray,
    max_iter: int = 50,
) -> float:
    """
    Scalar temperature scaling fitted on val by minimizing NLL.
    Returns T >= 1e-3
    """
    if logits_np.size == 0:
        return 1.0

    logits = torch.from_numpy(logits_np).to(device=device, dtype=torch.float32)
    y_true = torch.from_numpy(y_true_np).to(device=device, dtype=torch.long)

    T = torch.ones((), device=device, dtype=torch.float32, requires_grad=True)

    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), y_true)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())

def metrics_from_logits_np(
    y_true: np.ndarray,
    logits_np: np.ndarray,
    temperature: float = 1.0,
    ece_bins: int = 15,
):
    if logits_np.size == 0:
        return {
            "val_acc": float("nan"),
            "val_macro_p": float("nan"),
            "val_macro_r": float("nan"),
            "val_macro_f1": float("nan"),
            "val_w_f1": float("nan"),
            "val_auc_macro_ovr": float("nan"),
            "val_mean_confidence": float("nan"),
            "val_ece_15bins": float("nan"),
            "val_nll": float("nan"),
            "val_brier": float("nan"),
        }

    logits_cal = logits_np / max(float(temperature), 1e-6)
    probs = softmax_np(logits_cal)

    y_pred = probs.argmax(axis=1)
    val_acc = float(accuracy_score(y_true, y_pred))

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    val_w_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    val_auc_macro_ovr = float(auc_macro_ovr(y_true, probs))

    conf = probs.max(axis=1)
    val_mean_confidence = float(np.mean(conf))

    val_nll = float(log_loss(y_true, probs, labels=np.arange(num_classes)))
    val_brier = float(brier_multiclass(y_true, probs, n_classes=num_classes))
    val_ece = float(ece_from_probs(y_true, probs, n_bins=ece_bins))

    return {
        "val_acc": val_acc,
        "val_macro_p": float(p_macro),
        "val_macro_r": float(r_macro),
        "val_macro_f1": float(f1_macro),
        "val_w_f1": val_w_f1,
        "val_auc_macro_ovr": val_auc_macro_ovr,
        "val_mean_confidence": val_mean_confidence,
        "val_ece_15bins": val_ece,   # computed with ece_bins (use 15 for paper)
        "val_nll": val_nll,
        "val_brier": val_brier,
    }

# -----------------------
# NEW: STANDARD (HARD) ECE METRIC FOR REPORTING (non-differentiable)
# -----------------------
@torch.inference_mode()
def compute_ece_metric(
    model,
    loader,
    cfg,
    pruned: bool,
    graph_runner=None,
    n_bins: int = 15,
    max_steps=None,
):
    model.eval()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    bin_count = torch.zeros(n_bins, device=device)
    bin_conf_sum = torch.zeros(n_bins, device=device)
    bin_acc_sum = torch.zeros(n_bins, device=device)

    total = 0
    steps = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if pruned:
            if graph_runner is not None:
                logits = graph_runner(xb)
            else:
                logits = hard_pruned_predict_batch_fast(
                    model, xb,
                    tau=cfg["tau_infer"],
                    Kmax=cfg["Kmax_infer"],
                    use_dynamic=cfg.get("dynamic_kmax", False),
                    print_escalation=False,
                    entropy_thr=cfg.get("entropy_thr", None),
                    margin_thr=cfg.get("margin_thr", 0.15),
                    per_sample_escalation=cfg.get("per_sample_escalation", True),
                )
        else:
            logits = model(
                xb, training=False,
                gate_temperature=cfg["gate_temperature"],
                target_K=cfg["target_K"],
                budget_weight=cfg["budget_weight"],
            )[0]

        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == yb).float()

        idx = torch.bucketize(conf, bin_edges, right=True) - 1
        idx = idx.clamp(0, n_bins - 1)

        ones = torch.ones_like(conf)
        bin_count.scatter_add_(0, idx, ones)
        bin_conf_sum.scatter_add_(0, idx, conf)
        bin_acc_sum.scatter_add_(0, idx, correct)

        total += int(xb.size(0))
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    if total <= 0:
        return 0.0

    nonzero = bin_count > 0
    avg_conf = torch.zeros_like(bin_conf_sum)
    avg_acc = torch.zeros_like(bin_acc_sum)
    avg_conf[nonzero] = bin_conf_sum[nonzero] / bin_count[nonzero]
    avg_acc[nonzero] = bin_acc_sum[nonzero] / bin_count[nonzero]

    ece = ((bin_count / float(total)) * (avg_acc - avg_conf).abs()).sum()
    return float(ece.item())

# -----------------------
# >>> NEW: CSV helpers + extra metrics from probs (no logic change) <<<
# -----------------------
def _append_row_csv(csv_path: str, fieldnames: list, row: dict):
    csv_path = str(csv_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(csv_path).exists()

    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        safe_row = {k: row.get(k, "") for k in fieldnames}
        w.writerow(safe_row)

@torch.inference_mode()
def collect_val_probs(
    model,
    loader,
    cfg,
    pruned: bool,
    graph_runner=None,
    max_steps=None,
):
    model.eval()
    y_true_list = []
    probs_list = []
    steps = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        if pruned:
            if graph_runner is not None:
                logits = graph_runner(xb)
            else:
                logits = hard_pruned_predict_batch_fast(
                    model, xb,
                    tau=cfg["tau_infer"],
                    Kmax=cfg["Kmax_infer"],
                    use_dynamic=cfg.get("dynamic_kmax", False),
                    print_escalation=False,
                    entropy_thr=cfg.get("entropy_thr", None),
                    margin_thr=cfg.get("margin_thr", 0.15),
                    per_sample_escalation=cfg.get("per_sample_escalation", True),
                )
        else:
            logits = model(
                xb, training=False,
                gate_temperature=cfg["gate_temperature"],
                target_K=cfg["target_K"],
                budget_weight=cfg["budget_weight"],
            )[0]

        probs = F.softmax(logits, dim=1).float().cpu().numpy()
        probs_list.append(probs)
        y_true_list.append(yb.numpy())

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    y_true = np.concatenate(y_true_list) if len(y_true_list) else np.array([], dtype=np.int64)
    probs = np.concatenate(probs_list) if len(probs_list) else np.zeros((0, num_classes), dtype=np.float32)
    return y_true, probs

def brier_multiclass(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    if probs.size == 0:
        return float("nan")
    oh = np.eye(n_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))

def auc_macro_ovr(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        if probs.size == 0:
            return float("nan")
        C = probs.shape[1]
        if C == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

# -----------------------
# STN GRID CACHE
# -----------------------
_GRID_CACHE = {}

def _grid_cache_key(patch_size: int, dev: torch.device, dtype: torch.dtype):
    dev_index = dev.index if (dev.type == "cuda" and dev.index is not None) else -1
    return (patch_size, dev.type, dev_index, dtype)

def get_base_grid(patch_size: int, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = _grid_cache_key(patch_size, dev, dtype)
    g = _GRID_CACHE.get(key, None)
    if g is None:
        lin = torch.linspace(-1.0, 1.0, patch_size, device=dev, dtype=dtype)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        base = torch.stack([xx, yy], dim=-1)
        base = base.view(1, 1, patch_size, patch_size, 2)
        _GRID_CACHE[key] = base
        g = base
    return g

# -----------------------
# STN CROP
# -----------------------
def stn_crop_all(images: torch.Tensor,
                 cx: torch.Tensor, cy: torch.Tensor, s: torch.Tensor,
                 patch_size: int) -> torch.Tensor:
    B, C, H, W = images.shape
    B2, K = cx.shape
    assert B2 == B and cy.shape == (B, K) and s.shape == (B, K)

    base = get_base_grid(patch_size, images.device, images.dtype)

    cx_n = cx * 2.0 - 1.0
    cy_n = cy * 2.0 - 1.0

    s_  = s.view(B, K, 1, 1, 1)
    cx_ = cx_n.view(B, K, 1, 1, 1)
    cy_ = cy_n.view(B, K, 1, 1, 1)

    grid = base * s_
    grid[..., 0:1] = grid[..., 0:1] + cx_
    grid[..., 1:2] = grid[..., 1:2] + cy_
    grid = grid.view(B * K, patch_size, patch_size, 2)

    imgs = images.unsqueeze(1).expand(B, K, C, H, W).contiguous().view(B * K, C, H, W)

    patches = F.grid_sample(
        imgs, grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True
    )
    return patches

# -----------------------
# ROI PREDICTOR (Depthwise separable)
# -----------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class ROIPredictor(nn.Module):
    def __init__(self, K: int, s_min: float, s_max: float, input_size: int = 224):
        super().__init__()
        self.K = K
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.input_size = input_size

        self.conv1 = DepthwiseSeparableConv(3, 24, 3, stride=2, padding=1)
        self.conv2 = DepthwiseSeparableConv(24, 48, 3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv(48, 96, 3, stride=2, padding=1)
        self.fc = nn.Linear(96, 128)

        self.roi_raw = nn.Linear(128, K * 3)
        self.gate_logits = nn.Linear(128, K)

    def forward(self, x_01: torch.Tensor):
        if x_01.shape[-1] != self.input_size:
            x_01 = F.interpolate(x_01, size=(self.input_size, self.input_size),
                                 mode='bilinear', align_corners=False)

        x = normalize_torch(x_01)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(2, 3))
        x = F.relu(self.fc(x))

        raw = self.roi_raw(x).view(-1, self.K, 3)
        cx = torch.sigmoid(raw[..., 0])
        cy = torch.sigmoid(raw[..., 1])
        s  = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(raw[..., 2])

        gate_logits = self.gate_logits(x)
        return cx, cy, s, gate_logits

# -----------------------
# MAIN MODEL
# -----------------------
class TrainSTNROIConvNeXt(nn.Module):
    def __init__(self, num_classes: int, K_rois: int, patch_size: int,
                 s_min: float, s_max: float,
                 backbone_name: str, roi_input_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.K_rois = K_rois
        self.K_total = K_rois + 1
        self.patch_size = int(patch_size)

        self.roi_net = ROIPredictor(K_rois, s_min=s_min, s_max=s_max,
                                    input_size=roi_input_size)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        self.feat_dim = getattr(self.backbone, "num_features", None)
        if self.feat_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.patch_size, self.patch_size)
                out = self._forward_backbone_features(dummy)
                self.feat_dim = out.shape[-1]

        self.attn_fc = nn.Linear(self.feat_dim, 1)
        self.cls_head = nn.Linear(self.feat_dim, num_classes)

        self.register_buffer("_cxg", torch.tensor(0.5), persistent=False)
        self.register_buffer("_cyg", torch.tensor(0.5), persistent=False)
        self.register_buffer("_sg",  torch.tensor(1.0), persistent=False)

    def _forward_backbone_features(self, z_norm: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            f = self.backbone.forward_features(z_norm)
        else:
            f = self.backbone(z_norm)
        if f.dim() == 4:
            f = f.mean(dim=(2, 3))
        return f

    def forward(self, x_01: torch.Tensor,
                training: bool,
                gate_temperature: float,
                target_K: float,
                budget_weight: float):
        B = x_01.shape[0]
        cx, cy, s, gate_logits = self.roi_net(x_01)

        probs = torch.sigmoid(gate_logits)
        expected_K = probs.sum(dim=1).mean()
        budget_loss = (expected_K - target_K) ** 2 * budget_weight

        gates = concrete_sigmoid(gate_logits, gate_temperature) if training else probs

        cxg = self._cxg.view(1, 1).expand(B, 1)
        cyg = self._cyg.view(1, 1).expand(B, 1)
        sg  = self._sg.view(1, 1).expand(B, 1)
        gg  = torch.ones((B, 1), device=x_01.device, dtype=gates.dtype)

        cx_all = torch.cat([cxg, cx], dim=1)
        cy_all = torch.cat([cyg, cy], dim=1)
        s_all  = torch.cat([sg,  s],  dim=1)
        g_all  = torch.cat([gg,  gates], dim=1)

        patches = stn_crop_all(x_01, cx_all, cy_all, s_all, patch_size=self.patch_size)
        z = normalize_torch(patches)

        feat = self._forward_backbone_features(z)
        feat = feat.view(B, self.K_total, -1)

        attn_logits = self.attn_fc(feat).squeeze(-1)
        attn_logits = attn_logits + torch.log(g_all.clamp(min=1e-6, max=1.0))
        attn = F.softmax(attn_logits, dim=1)

        pooled = (feat * attn.unsqueeze(-1)).sum(dim=1)
        logits = self.cls_head(pooled)
        return logits, budget_loss, expected_K

# -----------------------
# EMA
# -----------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# -----------------------
# FREEZE / UNFREEZE
# -----------------------
def set_backbone_trainable(model: TrainSTNROIConvNeXt, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    model.backbone.train(trainable)

# -----------------------
# COSINE ANNEALING + WARMUP
# -----------------------
class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            denom = max(1, (self.total_epochs - self.warmup_epochs))
            progress = (epoch - self.warmup_epochs) / denom
            progress = float(np.clip(progress, 0.0, 1.0))
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# -----------------------
# TRAIN / EVAL
# -----------------------
def train_one_epoch(model, loader, optimizer, cfg, ema=None, use_mixup=True):
    model.train()
    total_loss = total_ce = total_budget = total_expK = total_ece = 0.0
    n = 0

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=cfg["label_smoothing"]
    )

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if use_mixup and cfg["mixup_alpha"] > 0:
            xb, yb_a, yb_b, lam = mixup_data(xb, yb, alpha=cfg["mixup_alpha"])
        else:
            yb_a = yb_b = yb
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)

        logits, budget_loss, expected_K = model(
            xb, training=True,
            gate_temperature=cfg["gate_temperature"],
            target_K=cfg["target_K"],
            budget_weight=cfg["budget_weight"],
        )

        if use_mixup and cfg["mixup_alpha"] > 0:
            ce = mixup_criterion(logits, yb_a, yb_b, lam, criterion)
        else:
            ce = criterion(logits, yb)

        # --- NEW: Soft-ECE calibration constraint (minimal integration) ---
        ece_loss = logits.sum() * 0.0
        if cfg.get("ece_weight", 0.0) > 0:
            if use_mixup and cfg["mixup_alpha"] > 0:
                e1 = soft_ece_loss_from_logits(
                    logits, yb_a,
                    n_bins=cfg.get("ece_bins", 15),
                    smoothing=cfg.get("ece_smoothing", 0.02),
                    temperature=cfg.get("ece_temperature", 1.0),
                )
                e2 = soft_ece_loss_from_logits(
                    logits, yb_b,
                    n_bins=cfg.get("ece_bins", 15),
                    smoothing=cfg.get("ece_smoothing", 0.02),
                    temperature=cfg.get("ece_temperature", 1.0),
                )
                ece_loss = lam * e1 + (1.0 - lam) * e2
            else:
                ece_loss = soft_ece_loss_from_logits(
                    logits, yb,
                    n_bins=cfg.get("ece_bins", 15),
                    smoothing=cfg.get("ece_smoothing", 0.02),
                    temperature=cfg.get("ece_temperature", 1.0),
                )

        loss = ce + budget_loss + cfg.get("ece_weight", 0.0) * ece_loss

        if not torch.isfinite(loss):
            print("[FATAL] Non-finite loss encountered. Stopping.")
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        if ema is not None:
            ema.update()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_budget += budget_loss.item() * bs
        total_expK += expected_K.item() * bs
        total_ece += float(ece_loss.detach().item()) * bs
        n += bs

    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "budget": total_budget / n,
        "ece": total_ece / n,
        "expected_K": total_expK / n
    }

@torch.inference_mode()
def val_loss(model, loader, cfg, max_steps=None):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.0)

    total = 0.0
    n = 0
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits, budget_loss, _ = model(
            xb, training=False,
            gate_temperature=cfg["gate_temperature"],
            target_K=cfg["target_K"],
            budget_weight=cfg["budget_weight"],
        )
        ce = criterion(logits, yb)
        loss = ce + budget_loss

        bs = xb.size(0)
        total += loss.item() * bs
        n += bs

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return total / max(n, 1)

@torch.inference_mode()
def evaluate_full(model, loader, cfg, max_steps=None):
    model.eval()
    y_true, y_pred = [], []
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        logits, _, _ = model(
            xb, training=False,
            gate_temperature=cfg["gate_temperature"],
            target_K=cfg["target_K"],
            budget_weight=cfg["budget_weight"],
        )
        pred = logits.argmax(dim=1).cpu().numpy()

        y_true.append(yb.numpy())
        y_pred.append(pred)

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return np.concatenate(y_true), np.concatenate(y_pred)

# ============================================================
# FAST PRUNED INFERENCE (optional dynamic, CUDA-graph-safe ROI path)
# ============================================================
@torch.inference_mode()
def hard_pruned_predict_batch_fast(
    model: TrainSTNROIConvNeXt,
    x_01: torch.Tensor,
    tau: float,
    Kmax: int,
    use_dynamic: bool = False,
    entropy_thr: Optional[float] = None,
    margin_thr: float = 0.15,
    print_escalation: bool = False,
    per_sample_escalation: bool = True,
):
    model.eval()
    B = x_01.size(0)
    Kmax = int(Kmax)

    if INFER_CHANNELS_LAST and x_01.is_cuda:
        x_01 = x_01.contiguous(memory_format=torch.channels_last)

    def _roi_path_logits(x_batch: torch.Tensor) -> torch.Tensor:
        Bb = x_batch.size(0)

        cx, cy, s, gate_logits = model.roi_net(x_batch)
        probs = torch.sigmoid(gate_logits)

        top_p, top_idx = torch.topk(probs, k=Kmax, dim=1, largest=True, sorted=True)

        cx_sel = torch.gather(cx, 1, top_idx)
        cy_sel = torch.gather(cy, 1, top_idx)
        s_sel  = torch.gather(s,  1, top_idx)

        cxg = model._cxg.view(1, 1).expand(Bb, 1)
        cyg = model._cyg.view(1, 1).expand(Bb, 1)
        sg  = model._sg.view(1, 1).expand(Bb, 1)

        cx_all = torch.cat([cxg, cx_sel], dim=1)
        cy_all = torch.cat([cyg, cy_sel], dim=1)
        s_all  = torch.cat([sg,  s_sel],  dim=1)

        patches = stn_crop_all(x_batch, cx_all, cy_all, s_all, patch_size=model.patch_size)
        z = normalize_torch(patches)

        with torch.cuda.amp.autocast(enabled=(use_amp_pruned_infer and device.type == "cuda")):
            feat = model._forward_backbone_features(z)

        feat = feat.view(Bb, 1 + Kmax, -1)

        lin_dtype = model.attn_fc.weight.dtype
        feat = feat.to(dtype=lin_dtype)

        g_sel = top_p.to(dtype=lin_dtype)
        g_sel = g_sel * (g_sel > tau).to(g_sel.dtype)

        sum_g = g_sel.sum(dim=1)
        all_zero = (sum_g <= 0).to(dtype=lin_dtype)  # [Bb]
        first = top_p[:, 0].to(dtype=lin_dtype)
        g0 = g_sel[:, 0]
        g_sel[:, 0] = g0 + all_zero * (first - g0)

        g_all = torch.cat(
            [torch.ones((Bb, 1), device=x_batch.device, dtype=lin_dtype), g_sel],
            dim=1
        )

        attn_logits = model.attn_fc(feat).squeeze(-1)
        attn_logits = attn_logits + torch.log(g_all.clamp(min=1e-6))
        attn = F.softmax(attn_logits, dim=1)

        pooled = (feat * attn.unsqueeze(-1)).sum(dim=1)
        logits = model.cls_head(pooled)
        return logits

    if use_dynamic and Kmax > 0:
        cxg = model._cxg.view(1, 1).expand(B, 1)
        cyg = model._cyg.view(1, 1).expand(B, 1)
        sg  = model._sg.view(1, 1).expand(B, 1)

        patches = stn_crop_all(x_01, cxg, cyg, sg, patch_size=model.patch_size)
        z = normalize_torch(patches)

        with torch.cuda.amp.autocast(enabled=(use_amp_pruned_infer and device.type == "cuda")):
            feat0 = model._forward_backbone_features(z)

        lin_dtype = model.attn_fc.weight.dtype
        feat0 = feat0.to(dtype=lin_dtype)
        logits0 = model.cls_head(feat0)

        probs0 = F.softmax(logits0, dim=1)
        ent = entropy_from_probs(probs0)
        mar = margin_from_probs(probs0)

        if entropy_thr is None:
            C = probs0.size(1)
            entropy_thr = float(0.4 * np.log(max(C, 2)))

        uncertain_mask = (ent > entropy_thr) | (mar < margin_thr)

        if print_escalation:
            esc_rate = uncertain_mask.float().mean().item()
            print(f"[DYN] esc_rate={esc_rate*100:.1f}% | "
                  f"ent(mean)={ent.mean().item():.3f} thr={float(entropy_thr):.3f} | "
                  f"mar(mean)={mar.mean().item():.3f} thr={margin_thr:.3f}")

        if not uncertain_mask.any():
            return logits0

        if per_sample_escalation:
            logits_out = logits0.clone()
            logits_u = _roi_path_logits(x_01[uncertain_mask])
            logits_out[uncertain_mask] = logits_u
            return logits_out

        return _roi_path_logits(x_01)

    if Kmax == 0:
        cxg = model._cxg.view(1, 1).expand(B, 1)
        cyg = model._cyg.view(1, 1).expand(B, 1)
        sg  = model._sg.view(1, 1).expand(B, 1)

        patches = stn_crop_all(x_01, cxg, cyg, sg, patch_size=model.patch_size)
        z = normalize_torch(patches)

        with torch.cuda.amp.autocast(enabled=(use_amp_pruned_infer and device.type == "cuda")):
            feat = model._forward_backbone_features(z)

        lin_dtype = model.attn_fc.weight.dtype
        feat = feat.to(dtype=lin_dtype)
        logits = model.cls_head(feat)
        return logits

    return _roi_path_logits(x_01)

# -----------------------
# CUDA GRAPH RUNNER (fixed batch/shape, dynamic_kmax must be False)
# -----------------------
class CUDAGraphRunner:
    def __init__(self, fn, example_x: torch.Tensor, warmup: int = 10):
        assert example_x.is_cuda
        self.fn = fn
        self.static_x = example_x.clone()
        self.static_out = None
        self.graph = torch.cuda.CUDAGraph()
        self.stream = torch.cuda.Stream()

        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                _ = self.fn(self.static_x)
        torch.cuda.synchronize()

        with torch.cuda.graph(self.graph):
            self.static_out = self.fn(self.static_x)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor):
        b = x.size(0)
        B = self.static_x.size(0)

        if b == B:
            self.static_x.copy_(x)
            self.graph.replay()
            return self.static_out

        if b < B:
            self.static_x[:b].copy_(x)
            self.static_x[b:].zero_()
            self.graph.replay()
            return self.static_out[:b]

        return self.fn(x)

@torch.inference_mode()
def evaluate_pruned_fast(model, loader, cfg, graph_runner: Optional[CUDAGraphRunner] = None, max_steps=None):
    model.eval()
    y_true, y_pred = [], []
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        if graph_runner is not None:
            logits = graph_runner(xb)
        else:
            logits = hard_pruned_predict_batch_fast(
                model, xb,
                tau=cfg["tau_infer"],
                Kmax=cfg["Kmax_infer"],
                use_dynamic=cfg.get("dynamic_kmax", False),
                print_escalation=cfg.get("print_escalation", False),
                entropy_thr=cfg.get("entropy_thr", None),
                margin_thr=cfg.get("margin_thr", 0.15),
                per_sample_escalation=cfg.get("per_sample_escalation", True),
            )

        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.append(yb.numpy())
        y_pred.append(pred)

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return np.concatenate(y_true), np.concatenate(y_pred)

# -----------------------
# TTA (optional)
# -----------------------
@torch.inference_mode()
def evaluate_pruned_tta(model, loader, cfg, n_augments=5, max_steps=None):
    model.eval()
    y_true, y_pred = [], []
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits_list = []

        logits = hard_pruned_predict_batch_fast(
            model, xb, tau=cfg["tau_infer"], Kmax=cfg["Kmax_infer"],
            use_dynamic=cfg.get("dynamic_kmax", False)
        )
        logits_list.append(logits)

        logits_flip = hard_pruned_predict_batch_fast(
            model, torch.flip(xb, dims=[-1]),
            tau=cfg["tau_infer"], Kmax=cfg["Kmax_infer"],
            use_dynamic=cfg.get("dynamic_kmax", False)
        )
        logits_list.append(logits_flip)

        if n_augments > 2:
            for angle in [5, -5]:
                from torchvision.transforms import functional as TF
                xb_rot = TF.rotate(xb, angle)
                logits_rot = hard_pruned_predict_batch_fast(
                    model, xb_rot, tau=cfg["tau_infer"], Kmax=cfg["Kmax_infer"],
                    use_dynamic=cfg.get("dynamic_kmax", False)
                )
                logits_list.append(logits_rot)

        logits_avg = torch.stack(logits_list).mean(dim=0)
        pred = logits_avg.argmax(dim=1).cpu().numpy()

        y_true.append(yb.numpy())
        y_pred.append(pred)

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return np.concatenate(y_true), np.concatenate(y_pred)

# -----------------------
# BENCHMARK
# -----------------------
@torch.inference_mode()
def benchmark_inference_val(model, loader, cfg, pruned: bool,
                            graph_runner: Optional[CUDAGraphRunner] = None,
                            warmup_batches: int = 10,
                            max_batches: Optional[int] = None):
    model.eval()

    w = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        if pruned:
            if graph_runner is not None:
                _ = graph_runner(xb)
            else:
                _ = hard_pruned_predict_batch_fast(
                    model, xb, tau=cfg["tau_infer"], Kmax=cfg["Kmax_infer"],
                    use_dynamic=cfg.get("dynamic_kmax", False)
                )
        else:
            _ = model(
                xb, training=False,
                gate_temperature=cfg["gate_temperature"],
                target_K=cfg["target_K"],
                budget_weight=cfg["budget_weight"],
            )[0]
        w += 1
        if w >= warmup_batches:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_seconds = 0.0
    total_images = 0
    total_batches = 0

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if pruned:
            if graph_runner is not None:
                _ = graph_runner(xb)
            else:
                _ = hard_pruned_predict_batch_fast(
                    model, xb, tau=cfg["tau_infer"], Kmax=cfg["Kmax_infer"],
                    use_dynamic=cfg.get("dynamic_kmax", False)
                )
        else:
            _ = model(
                xb, training=False,
                gate_temperature=cfg["gate_temperature"],
                target_K=cfg["target_K"],
                budget_weight=cfg["budget_weight"],
            )[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_seconds += (t1 - t0)
        total_images += int(xb.size(0))
        total_batches += 1

        if max_batches is not None and total_batches >= max_batches:
            break

    avg_ms_per_image = (total_seconds * 1000.0) / max(total_images, 1)
    avg_ms_per_batch = (total_seconds * 1000.0) / max(total_batches, 1)
    throughput = total_images / max(total_seconds, 1e-12)

    return {
        "warmup_batches": int(warmup_batches),
        "amp_used": bool(pruned and use_amp_pruned_infer and device.type == "cuda"),
        "total_images": int(total_images),
        "total_batches": int(total_batches),
        "total_seconds": float(total_seconds),
        "avg_ms_per_image": float(avg_ms_per_image),
        "avg_ms_per_batch": float(avg_ms_per_batch),
        "throughput_img_per_s": float(throughput),
    }

# -----------------------
# TRAINING RUNNER
# -----------------------
def run_training(cfg):
    model = TrainSTNROIConvNeXt(
        num_classes=num_classes,
        K_rois=cfg["K_rois"],
        patch_size=cfg["patch_size"],
        s_min=cfg["s_min"],
        s_max=cfg["s_max"],
        backbone_name=cfg["backbone_name"],
        roi_input_size=cfg["roi_input_size"],
    ).to(device)

    print("\nModel created. Feature dim:", model.feat_dim)
    print("Backbone:", cfg["backbone_name"])
    print("ROI Predictor Input Size:", cfg["roi_input_size"])
    print("K_rois:", cfg["K_rois"])
    print(f"Calib constraint: ece_weight={cfg.get('ece_weight',0.0)} | bins={cfg.get('ece_bins',15)} | smooth={cfg.get('ece_smoothing',0.02)}")

    os.makedirs(os.path.dirname(CKPT_SAVE_PATH), exist_ok=True)

    # ---- Stage 1 ----
    print("\n=== Stage 1: Train head+ROI+gates (backbone frozen) ===")
    set_backbone_trainable(model, trainable=False)

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_stage1"], eps=1e-7, weight_decay=0.01
    )
    sched1 = CosineAnnealingWarmup(
        opt1, warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["stage1_epochs"], min_lr=1e-7
    )

    best_val_1 = float("inf")
    best_state_1 = None
    bad = 0

    for epoch in range(1, cfg["stage1_epochs"] + 1):
        lr = sched1.step(epoch - 1)
        tr = train_one_epoch(model, train_loader, opt1, cfg, ema=None, use_mixup=False)
        if tr is None:
            break
        vl = val_loss(model, val_loader, cfg, max_steps=None)

        print(
            f"Epoch {epoch}/{cfg['stage1_epochs']} | "
            f"loss={tr['loss']:.4f} (ce={tr['ce']:.4f} budget={tr['budget']:.4f} ece={tr['ece']:.4f}) | "
            f"expected_K={tr['expected_K']:.3f} | val_loss={vl:.4f} | lr={lr:.2e}"
        )

        if vl < best_val_1 - 1e-6:
            best_val_1 = vl
            best_state_1 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("Early stopping (stage 1).")
                break

    if best_state_1 is not None:
        model.load_state_dict(best_state_1)

    # ---- Stage 2 ----
    print("\n=== Stage 2: Fine-tune full model (backbone unfrozen) ===")
    set_backbone_trainable(model, trainable=True)

    opt2 = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr_stage2"], eps=1e-7, weight_decay=0.01
    )
    sched2 = CosineAnnealingWarmup(
        opt2, warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"], min_lr=1e-7
    )

    ema = ModelEMA(model, decay=cfg["ema_decay"])

    best_val_2 = float("inf")
    best_state_2 = None
    bad = 0

    # >>> NEW: best epoch tracking (no logic change) <<<
    best_epoch_2 = -1
    best_by_2 = "stage2_val_loss(EMA)"

    for epoch in range(cfg["stage1_epochs"] + 1, cfg["epochs"] + 1):
        lr = sched2.step(epoch - 1)
        tr = train_one_epoch(model, train_loader, opt2, cfg, ema=ema, use_mixup=True)
        if tr is None:
            break

        ema.apply_shadow()
        vl = val_loss(model, val_loader, cfg, max_steps=None)
        ema.restore()

        print(f"Epoch {epoch}/{cfg['epochs']} | "
              f"loss={tr['loss']:.4f} (ce={tr['ce']:.4f} budget={tr['budget']:.4f} ece={tr['ece']:.4f}) | "
              f"expected_K={tr['expected_K']:.3f} | val_loss={vl:.4f} | lr={lr:.2e}")

        if vl < best_val_2 - 1e-6:
            best_val_2 = vl
            best_epoch_2 = epoch  # <<< NEW
            ema.apply_shadow()
            best_state_2 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()

            torch.save(best_state_2, CKPT_SAVE_PATH)
            print(f"[SAVED] best stage-2 EMA weights -> {CKPT_SAVE_PATH}")
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("Early stopping (stage 2).")
                break

    if best_state_2 is not None:
        model.load_state_dict(best_state_2)
    else:
        ema.apply_shadow()
        torch.save(model.state_dict(), CKPT_SAVE_PATH)
        ema.restore()
        print(f"[SAVED] (fallback) last EMA weights -> {CKPT_SAVE_PATH}")

    # >>> NEW: attach best info to model (no signature change) <<<
    model._best_epoch = int(best_epoch_2)
    model._best_by = str(best_by_2)
    model._best_val = float(best_val_2) if np.isfinite(best_val_2) else float("nan")

    return model

# -----------------------
# MAIN
# -----------------------
def main():
    cfg = {
        "backbone_name": BACKBONE_NAME,
        "patch_size": patch_size,
        "roi_input_size": roi_predictor_size,
        "s_min": s_min,
        "s_max": s_max,

        "K_rois": K_rois,

        "epochs": epochs,
        "stage1_epochs": stage1_epochs,
        "warmup_epochs": warmup_epochs,

        "lr_stage1": lr_stage1,
        "lr_stage2": lr_stage2,

        "target_K": target_K,
        "budget_weight": budget_weight,
        "gate_temperature": gate_temperature,

        "tau_infer": tau_infer,
        "Kmax_infer": Kmax_infer,
        "dynamic_kmax": dynamic_kmax,

        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
        "ema_decay": ema_decay,

        "entropy_thr": None,
        "margin_thr": 0.15,
        "per_sample_escalation": True,
        "print_escalation": True,

        "patience": 15,

        # training-time calibration constraint (optional)
        "ece_weight": ece_weight,
        "ece_bins": ece_bins,
        "ece_smoothing": ece_smoothing,
        "ece_temperature": ece_temperature,
    }

    print("\n" + "=" * 70)
    print("🧠 OPTIMIZED BRAIN TUMOR MRI CLASSIFIER (RTX 3080)")
    print("Apple-to-apple batch=16")
    print("=" * 70)
    for k in [
        "K_rois", "Kmax_infer", "dynamic_kmax", "tau_infer",
        "gate_temperature", "target_K", "budget_weight",
        "lr_stage1", "lr_stage2",
        "ece_weight", "ece_bins", "ece_smoothing", "ece_temperature"
    ]:
        print(f"  {k}: {cfg[k]}")
    print("=" * 70)

    # ---------------- TRAIN ----------------
    model = run_training(cfg)
    model.eval()

    # channels_last for inference
    if INFER_CHANNELS_LAST and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # torch.compile
    if device.type == "cuda":
        try:
            print("\n⚡ Compiling model with torch.compile(mode='max-autotune-no-cudagraphs')...")
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            print("✅ torch.compile enabled")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")

    # CUDA Graph for PRUNED inference (only when dynamic_kmax=False)
    graph_runner = None
    if device.type == "cuda" and (not cfg.get("dynamic_kmax", False)):
        xb0, _ = next(iter(val_loader_fast))
        xb0 = xb0.to(device, non_blocking=True)
        if INFER_CHANNELS_LAST:
            xb0 = xb0.contiguous(memory_format=torch.channels_last)

        def pruned_fn(xb):
            return hard_pruned_predict_batch_fast(
                model, xb,
                tau=cfg["tau_infer"],
                Kmax=cfg["Kmax_infer"],
                use_dynamic=False,
                entropy_thr=cfg.get("entropy_thr", None),
                margin_thr=cfg.get("margin_thr", 0.15),
                print_escalation=False,
                per_sample_escalation=True,
            )

        try:
            graph_runner = CUDAGraphRunner(pruned_fn, xb0, warmup=10)
            print("✅ CUDA Graph enabled for PRUNED inference (fixed path).")
        except Exception as e:
            graph_runner = None
            print(f"⚠️  CUDA Graph capture failed (continuing without graphs): {e}")

    # ============================================================
    # TEMPERATURE SCALING (post-hoc) — calibrate FULL and PRUNED separately
    # ============================================================
    print("\n" + "=" * 70)
    print("🌡️  POST-HOC CALIBRATION (Temperature Scaling) on VAL")
    print("=" * 70)

    # FULL logits on val
    y_true_full_ts, logits_full_ts = collect_val_logits(
        model, val_loader_fast, cfg, pruned=False, graph_runner=None, max_steps=None
    )
    T_full = fit_temperature_on_val(logits_full_ts, y_true_full_ts, max_iter=50)

    # PRUNED logits on val
    y_true_pr_ts, logits_pr_ts = collect_val_logits(
        model, val_loader_fast, cfg, pruned=True, graph_runner=graph_runner, max_steps=None
    )
    T_pruned = fit_temperature_on_val(logits_pr_ts, y_true_pr_ts, max_iter=50)

    print(f"Learned temperature:  T_full={T_full:.4f} | T_pruned={T_pruned:.4f}")
    print("=" * 70)

    # Uncalibrated vs calibrated metrics (paper-friendly)
    m_full_uncal = metrics_from_logits_np(y_true_full_ts, logits_full_ts, temperature=1.0, ece_bins=15)
    m_full_cal   = metrics_from_logits_np(y_true_full_ts, logits_full_ts, temperature=T_full, ece_bins=15)

    m_pr_uncal   = metrics_from_logits_np(y_true_pr_ts, logits_pr_ts, temperature=1.0, ece_bins=15)
    m_pr_cal     = metrics_from_logits_np(y_true_pr_ts, logits_pr_ts, temperature=T_pruned, ece_bins=15)

    print("\nFULL (uncal)  : "
          f"ECE15={m_full_uncal['val_ece_15bins']:.4f} | "
          f"NLL={m_full_uncal['val_nll']:.4f} | "
          f"Brier={m_full_uncal['val_brier']:.4f}")

    print("FULL (calib)  : "
          f"ECE15={m_full_cal['val_ece_15bins']:.4f} | "
          f"NLL={m_full_cal['val_nll']:.4f} | "
          f"Brier={m_full_cal['val_brier']:.4f}")

    print("\nPRUNED (uncal) : "
          f"ECE15={m_pr_uncal['val_ece_15bins']:.4f} | "
          f"NLL={m_pr_uncal['val_nll']:.4f} | "
          f"Brier={m_pr_uncal['val_brier']:.4f}")

    print("PRUNED (calib) : "
          f"ECE15={m_pr_cal['val_ece_15bins']:.4f} | "
          f"NLL={m_pr_cal['val_nll']:.4f} | "
          f"Brier={m_pr_cal['val_brier']:.4f}")

    # ---------------- EVAL FULL (labels) ----------------
    print("\n" + "=" * 70)
    print("📊 FULL MODEL EVALUATION (accuracy/CM unaffected by TS)")
    print("=" * 70)

    y_true, y_pred = evaluate_full(model, val_loader, cfg, max_steps=None)
    acc_full = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc_full:.4f} ({acc_full*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ---------------- EVAL PRUNED (labels) ----------------
    print("\n" + "=" * 70)
    print("⚡ PRUNED MODEL EVALUATION (accuracy/CM unaffected by TS)")
    print("=" * 70)

    y_true_p, y_pred_p = evaluate_pruned_fast(model, val_loader_fast, cfg, graph_runner=graph_runner, max_steps=None)
    acc_pruned = accuracy_score(y_true_p, y_pred_p)
    print(f"Accuracy: {acc_pruned:.4f} ({acc_pruned*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_true_p, y_pred_p, target_names=class_names, digits=4, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_p, y_pred_p))

    # ---------------- BENCHMARK PRUNED ----------------
    print("\n" + "=" * 70)
    print("⚡ INFERENCE BENCHMARK [PRUNED - OPTIMIZED]")
    print("=" * 70)

    bench_pruned = benchmark_inference_val(
        model, val_loader_fast, cfg, pruned=True,
        graph_runner=graph_runner,
        warmup_batches=20, max_batches=None
    )

    print(f"Warmup batches          : {bench_pruned['warmup_batches']}")
    print(f"Batch size              : {batch_size_inference}")
    print(f"AMP enabled             : {bench_pruned['amp_used']}")
    print(f"Total images processed  : {bench_pruned['total_images']}")
    print(f"Total batches           : {bench_pruned['total_batches']}")
    print(f"Total inference time    : {bench_pruned['total_seconds']:.6f} s")
    print(f"Avg time per image      : {bench_pruned['avg_ms_per_image']:.4f} ms")
    print(f"Avg time per batch      : {bench_pruned['avg_ms_per_batch']:.4f} ms")
    print(f"\n🚀 THROUGHPUT           : {bench_pruned['throughput_img_per_s']:.2f} images/s")
    print("=" * 70)

    # FULL benchmark (optional, for CSV)
    bench_full = benchmark_inference_val(
        model, val_loader_fast, cfg, pruned=False,
        graph_runner=None,
        warmup_batches=20, max_batches=None
    )

    # ============================================================
    # CSV LOGGING (calibrated metrics)
    # ============================================================
    modality = Path(train_dir).name
    base_model_name = f"STNROI+{BACKBONE_NAME}"
    params_m = float(sum(p.numel() for p in model.parameters()) / 1e6)
    best_epoch = int(getattr(model, "_best_epoch", -1))
    best_by = str(getattr(model, "_best_by", "stage2_val_loss(EMA)"))

    fieldnames = [
        "modality","model","best_epoch","best_by",
        "val_acc","val_macro_p","val_macro_r","val_macro_f1","val_w_f1",
        "val_auc_macro_ovr","val_mean_confidence","val_ece_15bins","val_nll","val_brier",
        "temp_T","params_m","ms_per_img","throughput_img_s"
    ]

    # FULL row (calibrated)
    row_full = {
        "modality": modality,
        "model": base_model_name + "|full|temp_scaled",
        "best_epoch": best_epoch,
        "best_by": best_by,

        **m_full_cal,
        "temp_T": float(T_full),

        "params_m": params_m,
        "ms_per_img": float(bench_full["avg_ms_per_image"]),
        "throughput_img_s": float(bench_full["throughput_img_per_s"]),
    }
    _append_row_csv(METRICS_CSV_PATH, fieldnames, row_full)

    # PRUNED row (calibrated)
    row_pruned = {
        "modality": modality,
        "model": base_model_name + f"|pruned(Kmax={cfg['Kmax_infer']},tau={cfg['tau_infer']},dyn={cfg['dynamic_kmax']})|temp_scaled",
        "best_epoch": best_epoch,
        "best_by": best_by,

        **m_pr_cal,
        "temp_T": float(T_pruned),

        "params_m": params_m,
        "ms_per_img": float(bench_pruned["avg_ms_per_image"]),
        "throughput_img_s": float(bench_pruned["throughput_img_per_s"]),
    }
    _append_row_csv(METRICS_CSV_PATH, fieldnames, row_pruned)

    print(f"\n🧾 CSV metrics appended (calibrated FULL + PRUNED) -> {METRICS_CSV_PATH}")

    # ---------------- SUMMARY ----------------
    print("\n" + "=" * 70)
    print("📈 FINAL RESULTS SUMMARY (Calibrated)")
    print("=" * 70)
    print(f"Full  Acc              : {m_full_cal['val_acc']*100:.2f}%")
    print(f"Full  ECE15 / NLL / Bri : {m_full_cal['val_ece_15bins']:.4f} / {m_full_cal['val_nll']:.4f} / {m_full_cal['val_brier']:.4f}   (T={T_full:.3f})")
    print(f"Pruned Acc             : {m_pr_cal['val_acc']*100:.2f}%")
    print(f"Pruned ECE15 / NLL / Bri: {m_pr_cal['val_ece_15bins']:.4f} / {m_pr_cal['val_nll']:.4f} / {m_pr_cal['val_brier']:.4f}   (T={T_pruned:.3f})")
    print(f"Inference Throughput   : {bench_pruned['throughput_img_per_s']:.2f} images/s")
    print("=" * 70)

    print(f"\n💾 Best model saved at: {CKPT_SAVE_PATH}")
    print("\n✨ Training complete!")

# ============================================================
# OPTUNA TUNING (ONLY THIS PART MODIFIED)
# ============================================================
TUNE_ENABLE_MULTI_OBJECTIVE = True
TUNE_N_TRIALS = 25

TUNE_FAST_MODE = True
TUNE_EPOCHS = 100
TUNE_STAGE1_EPOCHS = 25
TUNE_PATIENCE = 12

TUNE_MAX_STEPS_EVAL = 60
TUNE_MAX_BENCH_BATCHES = 80
TUNE_WARMUP_BATCHES = 10

TUNE_COMPILE_FOR_BENCH = True

def _default_cfg_dict():
    return {
        "backbone_name": BACKBONE_NAME,
        "patch_size": patch_size,
        "roi_input_size": roi_predictor_size,
        "s_min": s_min,
        "s_max": s_max,

        "K_rois": K_rois,

        "epochs": epochs,
        "stage1_epochs": stage1_epochs,
        "warmup_epochs": warmup_epochs,

        "lr_stage1": lr_stage1,
        "lr_stage2": lr_stage2,

        "target_K": target_K,
        "budget_weight": budget_weight,
        "gate_temperature": gate_temperature,

        "tau_infer": tau_infer,
        "Kmax_infer": Kmax_infer,
        "dynamic_kmax": dynamic_kmax,

        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
        "ema_decay": ema_decay,

        "entropy_thr": None,
        "margin_thr": 0.15,
        "per_sample_escalation": True,
        "print_escalation": False,

        "patience": 15,

        "ece_weight": ece_weight,
        "ece_bins": ece_bins,
        "ece_smoothing": ece_smoothing,
        "ece_temperature": ece_temperature,
    }

def _cfg_from_trial(trial):
    cfg = _default_cfg_dict()

    if TUNE_FAST_MODE:
        cfg["epochs"] = TUNE_EPOCHS
        cfg["stage1_epochs"] = TUNE_STAGE1_EPOCHS
        cfg["patience"] = TUNE_PATIENCE

    # ✅ Tune ONLY calibration parameters
    cfg["ece_weight"] = trial.suggest_float("ece_weight", 0.0, 0.08)  # 0 disables it
    cfg["ece_bins"] = trial.suggest_categorical("ece_bins", [10, 15, 20, 25])
    cfg["ece_smoothing"] = trial.suggest_float("ece_smoothing", 0.005, 0.08, log=True)
    cfg["ece_temperature"] = trial.suggest_float("ece_temperature", 0.7, 2.0)

    return cfg

def _pick_best_pareto(
    best_trials,
    target_acc=0.98,
    target_thr=2300.0,
    target_ece=0.02,
    target_nll=0.25,
    target_brier=0.10,
    w_cal=1.0,
):
    """
    Picks one trial from Pareto set by minimizing a soft distance to targets.
    values = (acc, thr, ece_mean, nll_mean, brier_mean)
    """
    best = None
    best_score = float("inf")

    for t in best_trials:
        acc, thr, ece, nll, brier = t.values

        da = max(0.0, float(target_acc) - float(acc)) / max(float(target_acc), 1e-12)
        dt = max(0.0, float(target_thr) - float(thr)) / max(float(target_thr), 1e-12)

        de = max(0.0, float(ece) - float(target_ece)) / max(float(target_ece), 1e-12)
        dn = max(0.0, float(nll) - float(target_nll)) / max(float(target_nll), 1e-12)
        db = max(0.0, float(brier) - float(target_brier)) / max(float(target_brier), 1e-12)

        score = (da * da) + (dt * dt) + float(w_cal) * ((de * de) + (dn * dn) + (db * db))

        if score < best_score:
            best_score = score
            best = t

    return best

def tune_main():
    import gc
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    # ---- floors to prevent Optuna "cheating" by selecting low-acc but well-calibrated trials ----
    ACC_FLOOR = 0.97     # set 0 to disable
    THR_FLOOR = 0.0      # set e.g. 800.0 if you want to enforce speed

    def objective(trial):
        cfg = _cfg_from_trial(trial)
        model = None

        try:
            set_seed(seed)
            model = run_training(cfg)
            model.eval()

            if INFER_CHANNELS_LAST and device.type == "cuda":
                model = model.to(memory_format=torch.channels_last)

            if TUNE_COMPILE_FOR_BENCH and device.type == "cuda":
                try:
                    model = torch.compile(model, mode="max-autotune-no-cudagraphs")
                except Exception:
                    pass

            # -------------------------
            # 1) PRUNED accuracy (labels)
            # -------------------------
            y_true_p, y_pred_p = evaluate_pruned_fast(
                model, val_loader_fast, cfg, graph_runner=None, max_steps=TUNE_MAX_STEPS_EVAL
            )
            acc_pruned = float(accuracy_score(y_true_p, y_pred_p))

            # -------------------------
            # 2) PRUNED throughput
            # -------------------------
            bench = benchmark_inference_val(
                model, val_loader_fast, cfg, pruned=True,
                graph_runner=None,
                warmup_batches=TUNE_WARMUP_BATCHES,
                max_batches=TUNE_MAX_BENCH_BATCHES
            )
            thr = float(bench["throughput_img_per_s"])

            # -------------------------
            # 3) CALIBRATION METRICS (FULL + PRUNED), both temperature-scaled
            # -------------------------
            y_full, logits_full = collect_val_logits(
                model, val_loader_fast, cfg, pruned=False, graph_runner=None, max_steps=TUNE_MAX_STEPS_EVAL
            )
            T_full = fit_temperature_on_val(logits_full, y_full, max_iter=50)
            m_full = metrics_from_logits_np(y_full, logits_full, temperature=T_full, ece_bins=15)

            y_pr, logits_pr = collect_val_logits(
                model, val_loader_fast, cfg, pruned=True, graph_runner=None, max_steps=TUNE_MAX_STEPS_EVAL
            )
            T_pr = fit_temperature_on_val(logits_pr, y_pr, max_iter=50)
            m_pr = metrics_from_logits_np(y_pr, logits_pr, temperature=T_pr, ece_bins=15)

            ece_mean   = 0.5 * (float(m_full["val_ece_15bins"]) + float(m_pr["val_ece_15bins"]))
            nll_mean   = 0.5 * (float(m_full["val_nll"])        + float(m_pr["val_nll"]))
            brier_mean = 0.5 * (float(m_full["val_brier"])      + float(m_pr["val_brier"]))

            # -------------------------
            # 4) Enforce floors via penalty (so bad-acc trials are dominated)
            # -------------------------
            if (ACC_FLOOR > 0 and acc_pruned < ACC_FLOOR) or (THR_FLOOR > 0 and thr < THR_FLOOR):
                ece_mean += 1e3
                nll_mean += 1e3
                brier_mean += 1e3

            # -------------------------
            # 5) Log helpful attrs
            # -------------------------
            trial.set_user_attr("acc_pruned", float(acc_pruned))
            trial.set_user_attr("throughput_img_s", float(thr))

            trial.set_user_attr("T_full", float(T_full))
            trial.set_user_attr("T_pruned", float(T_pr))

            trial.set_user_attr("ece_full_cal", float(m_full["val_ece_15bins"]))
            trial.set_user_attr("nll_full_cal", float(m_full["val_nll"]))
            trial.set_user_attr("brier_full_cal", float(m_full["val_brier"]))

            trial.set_user_attr("ece_pruned_cal", float(m_pr["val_ece_15bins"]))
            trial.set_user_attr("nll_pruned_cal", float(m_pr["val_nll"]))
            trial.set_user_attr("brier_pruned_cal", float(m_pr["val_brier"]))

            trial.set_user_attr("ece_mean", float(ece_mean))
            trial.set_user_attr("nll_mean", float(nll_mean))
            trial.set_user_attr("brier_mean", float(brier_mean))

            # -------------------------
            # 6) Return objectives
            # -------------------------
            if TUNE_ENABLE_MULTI_OBJECTIVE:
                # 5-objective: maximize acc/thr, minimize calibration errors
                return acc_pruned, thr, ece_mean, nll_mean, brier_mean
            else:
                # single scalar (optional): prioritize acc & speed, penalize calibration
                return float(acc_pruned + 1e-4 * thr - 0.25 * ece_mean - 0.02 * nll_mean - 0.02 * brier_mean)

        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_warmup_steps=5)

    if TUNE_ENABLE_MULTI_OBJECTIVE:
        # ✅ UPDATED: 5-objective directions
        study = optuna.create_study(
            directions=["maximize", "maximize", "minimize", "minimize", "minimize"],
            sampler=sampler,
            pruner=pruner,
            study_name="stn_roi_convnextv2_tuning",
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="stn_roi_convnextv2_tuning",
        )

    print("\n" + "=" * 70)
    print(f"🔍 OPTUNA TUNING START | trials={TUNE_N_TRIALS} | multi_objective={TUNE_ENABLE_MULTI_OBJECTIVE}")
    if TUNE_ENABLE_MULTI_OBJECTIVE:
        print("Objectives: maximize(acc_pruned), maximize(thr), minimize(ece_mean), minimize(nll_mean), minimize(brier_mean)")
    print("=" * 70)

    study.optimize(objective, n_trials=TUNE_N_TRIALS, gc_after_trial=True, show_progress_bar=True)

    print("\n" + "=" * 70)
    print("✅ OPTUNA DONE")
    print("=" * 70)

    if TUNE_ENABLE_MULTI_OBJECTIVE:
        print(f"Pareto trials: {len(study.best_trials)}")
        best = _pick_best_pareto(
            study.best_trials,
            target_acc=0.98,
            target_thr=2300.0,
            target_ece=0.02,
            target_nll=0.25,
            target_brier=0.10,
            w_cal=1.0,
        )
        print("\nChosen best (closest to targets):")
        print("  values(acc, thr, ece_mean, nll_mean, brier_mean):", best.values)
        print("  params:", best.params)
        print("  user_attrs(ece_full_cal, ece_pruned_cal):",
              best.user_attrs.get("ece_full_cal", None), best.user_attrs.get("ece_pruned_cal", None))
    else:
        print("Best value:", study.best_value)
        print("Best params:", study.best_params)

    try:
        import pandas as pd
        df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
        df.to_csv("optuna_trials.csv", index=False)
        print("Saved: optuna_trials.csv")
    except Exception as e:
        print("CSV save skipped:", e)

# -----------------------
# ENTRY
# -----------------------
import sys as _sys
if __name__ == "__main__" and ("--tune" in _sys.argv):
    tune_main()
    raise SystemExit(0)

if __name__ == "__main__":
    main()
