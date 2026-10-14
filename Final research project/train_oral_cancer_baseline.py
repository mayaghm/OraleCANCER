
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(np.float32)
    y_prob = np.clip(y_prob.astype(np.float32), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]

        if i == 0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob > left) & (y_prob <= right)

        if mask.any():
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += float(np.abs(acc - conf) * mask.mean())

    return float(ece)


def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, 201)
    best_t, best_j = 0.5, -1e9

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best_t = float(t)

    return best_t, float(best_j)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)

    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10bins": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
    }


LABEL_MAP_DEFAULT = {
    "cancer": 1,
    "malignant": 1,
    "high-risk": 1,
    "high_risk": 1,
    "oc": 1,
    "oca": 1,
    "oscc": 1,
    "non-cancer": 0,
    "non_cancer": 0,
    "benign": 0,
    "low-risk": 0,
    "low_risk": 0,
    "normal": 0,
    "healthy": 0,
    "opmd": 0,
}


def normalize_label(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)

    if isinstance(x, float) and not math.isnan(x):
        return int(x)

    s = str(x).strip().lower()
    if s in LABEL_MAP_DEFAULT:
        return int(LABEL_MAP_DEFAULT[s])

    try:
        return int(float(s))
    except Exception as e:
        raise ValueError(f"Unknown label value: {x!r}") from e


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_path", "label"}

    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {sorted(required)}. Got {sorted(df.columns)}")

    df = df.copy()
    df["label"] = df["label"].apply(normalize_label).astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("This script supports binary labels only (0/1).")

    return df


def patient_level_stratified_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if patient_col not in df.columns:
        train_df, temp_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[label_col]
        )

        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=seed, stratify=temp_df[label_col]
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    patient_labels = (
        df.groupby(patient_col)[label_col]
        .apply(lambda s: int(s.value_counts().idxmax()))
        .reset_index()
        .rename(columns={label_col: "patient_label"})
    )

    inconsistent = df.groupby(patient_col)[label_col].nunique()
    n_inconsistent = int((inconsistent > 1).sum())
    if n_inconsistent > 0:
        print(f"[WARN] {n_inconsistent} patients have >1 label. Using majority label.")

    pats = patient_labels[patient_col].values
    ypat = patient_labels["patient_label"].values

    train_pats, test_pats = train_test_split(
        pats, test_size=test_size, random_state=seed, stratify=ypat
    )

    patient_labels_train = patient_labels[patient_labels[patient_col].isin(train_pats)]
    pats2 = patient_labels_train[patient_col].values
    ypat2 = patient_labels_train["patient_label"].values
    val_frac = val_size / (1.0 - test_size)

    train_pats2, val_pats = train_test_split(
        pats2, test_size=val_frac, random_state=seed, stratify=ypat2
    )

    train_df = df[df[patient_col].isin(train_pats2)]
    val_df = df[df[patient_col].isin(val_pats)]
    test_df = df[df[patient_col].isin(test_pats)]

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class OralImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: Optional[str], transform):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root) if img_root else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        pth = Path(p)
        if pth.is_absolute():
            return pth
        if self.img_root is None:
            return pth
        return self.img_root / pth

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row["image_path"]))
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        return x, label, str(img_path)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        return m

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 1)
        return m

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=weights)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, 1)
        return m

    if name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=weights)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, 1)
        return m

    raise ValueError("Unknown model name")


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    y_true, y_prob, paths = [], [], []

    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        y_true.append(y.numpy())
        y_prob.append(prob)
        paths.extend(list(p))

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    return y_true, y_prob, paths


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    running = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.detach().cpu().item()) * x.size(0)

    return running / len(loader.dataset)


@dataclass
class TrainConfig:
    data_csv: str
    img_root: Optional[str]
    out_dir: str
    model: str
    epochs: int
    batch_size: int
    lr: float
    wd: float
    img_size: int
    seed: int
    num_workers: int
    amp: bool
    pretrained: bool
    test_size: float
    val_size: float


def main_train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    df = load_dataframe(cfg.data_csv)

    train_df, val_df, test_df = patient_level_stratified_split(
        df,
        seed=cfg.seed,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        patient_col="patient_id",
        label_col="label",
    )

    print(f"[INFO] Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    pos_train = int(train_df["label"].sum())
    neg_train = int((train_df["label"] == 0).sum())
    print(f"[INFO] Train class counts: pos={pos_train} neg={neg_train}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = OralImageDataset(train_df, cfg.img_root, train_tf)
    val_ds = OralImageDataset(val_df, cfg.img_root, test_tf)
    test_ds = OralImageDataset(test_df, cfg.img_root, test_tf)

    y_train = train_df["label"].values.astype(int)
    class_counts = np.bincount(y_train, minlength=2)

    pos_weight = float(class_counts[0] / max(class_counts[1], 1))
    print(f"[INFO] pos_weight for BCEWithLogitsLoss = {pos_weight:.4f}")

    weights = np.where(y_train == 1, float(class_counts[0] / max(class_counts[1], 1)), 1.0)
    sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg.model, pretrained=cfg.pretrained).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auc = -1.0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp=cfg.amp)

        yv_true, yv_prob, _ = run_eval(model, val_loader, device)
        val_auc = float(roc_auc_score(yv_true, yv_prob)) if len(np.unique(yv_true)) == 2 else float("nan")

        scheduler.step(val_auc)

        val_thr, val_j = find_best_threshold_youden(yv_true, yv_prob)
        val_metrics = compute_binary_metrics(yv_true, yv_prob, threshold=val_thr)

        epoch_rec = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(val_auc),
            "val_threshold": float(val_thr),
            "val_youdenJ": float(val_j),
            "val_sensitivity": float(val_metrics["sensitivity"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_f1": float(val_metrics["f1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "sec": float(time.time() - t0),
        }
        history.append(epoch_rec)

        print(
            f"[E{epoch:03d}] loss={train_loss:.4f} val_auc={val_auc:.4f} "
            f"val_sens={val_metrics['sensitivity']:.3f} val_spec={val_metrics['specificity']:.3f} "
            f"val_f1={val_metrics['f1']:.3f} thr={val_thr:.3f} lr={epoch_rec['lr']:.2e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            ckpt = {
                "model_name": cfg.model,
                "state_dict": model.state_dict(),
                "val_auc": best_auc,
                "val_threshold": float(val_thr),
                "config": asdict(cfg),
            }
            torch.save(ckpt, best_path)
            print(f"[INFO] Saved new best checkpoint to: {best_path} (val_auc={best_auc:.4f})")

    ckpt = torch.load(best_path, map_location=device)
    model = build_model(ckpt["model_name"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    test_thr = float(ckpt.get("val_threshold", 0.5))
    yt_true, yt_prob, yt_paths = run_eval(model, test_loader, device)
    test_metrics = compute_binary_metrics(yt_true, yt_prob, threshold=test_thr)

    pred_df = pd.DataFrame({
        "image_path": yt_paths,
        "y_true": yt_true.astype(int),
        "y_prob": yt_prob.astype(float),
        "y_pred": (yt_prob >= test_thr).astype(int),
    })
    pred_df.to_csv(out_dir / "predictions_test.csv", index=False)

    report = {
        "best_val_auc": float(best_auc),
        "test_metrics": test_metrics,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "history": history,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[RESULT] Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"[INFO] Artifacts saved in: {out_dir}")


def main_predict(img_path: str, checkpoint: str, model_name_override: Optional[str], img_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    model_name = model_name_override or ckpt["model_name"]

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit).item()

    thr = float(ckpt.get("val_threshold", 0.5))
    pred = int(prob >= thr)

    print(f"[PRED] image={img_path}")
    print(f"[PRED] prob_cancer={prob:.4f} threshold={thr:.3f} => pred={pred} (1=cancer,0=non-cancer)")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_csv", type=str, default=None)
    ap.add_argument("--img_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="runs/exp")

    ap.add_argument("--model", type=str, default="efficientnet_b0",
                    choices=["resnet50", "efficientnet_b0", "densenet121", "mobilenet_v3_large"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)

    ap.add_argument("--predict", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--model_override", type=str, default=None)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict is not None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --predict")
        main_predict(args.predict, args.checkpoint, args.model_override, args.img_size)
    else:
        if args.data_csv is None:
            raise ValueError("--data_csv is required for training")

        cfg = TrainConfig(
            data_csv=args.data_csv,
            img_root=args.img_root,
            out_dir=args.out_dir,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            wd=args.wd,
            img_size=args.img_size,
            seed=args.seed,
            num_workers=args.num_workers,
            amp=bool(args.amp),
            pretrained=not bool(args.no_pretrained),
            test_size=float(args.test_size),
            val_size=float(args.val_size),
        )

        main_train(cfg)
