from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config  # noqa: E402
from data_loader import HAM10000Dataset  # noqa: E402
from federated_utils import build_model, build_transforms  # noqa: E402
from split_utils import lesion_level_split_indices  # noqa: E402


def _num_workers() -> int:
    env = os.getenv("EVAL_NUM_WORKERS")
    if env is not None:
        return int(env)
    return config.default_num_workers()


def build_splits():
    dataset = HAM10000Dataset(
        csv_path=config.METADATA_CSV,
        image_dir=config.IMAGES_DIR,
        transform=build_transforms(),
    )
    train_idx, val_idx, test_idx = lesion_level_split_indices(
        dataset.df,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.SPLIT_SEED,
    )
    _train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)
    return val_ds, test_ds


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for images, labels in tqdm(loader, desc="infer", leave=False):
        images = images.to(device, non_blocking=True)
        probs = F.softmax(model(images), dim=1)[:, 1]
        ys.append(labels.numpy())
        ps.append(probs.cpu().numpy())
    y_true = np.concatenate(ys)
    p_mal = np.concatenate(ps)
    return y_true, p_mal


def metrics_at_threshold(y_true, p_mal, th):
    y_pred = (p_mal >= th).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "malign_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "malign_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_threshold(y_true, p_mal):
    best = None
    recall_floor = float(os.getenv("RECALL_FLOOR", "0.85"))
    precision_floor = float(os.getenv("PRECISION_FLOOR", "0.30"))
    for th in np.arange(0.05, 0.96, 0.01):
        m = metrics_at_threshold(y_true, p_mal, float(th))
        if m["malign_recall"] < recall_floor or m["malign_precision"] < precision_floor:
            continue
        score = (
            m["macro_f1"],
            m["balanced_accuracy"],
            m["malign_recall"],
            m["malign_precision"],
        )
        if best is None or score > best["score"]:
            best = {"threshold": float(th), "metrics": m, "score": score}
    if best is None:
        for th in np.arange(0.05, 0.96, 0.01):
            m = metrics_at_threshold(y_true, p_mal, float(th))
            score = (m["macro_f1"], m["balanced_accuracy"], m["malign_recall"])
            if best is None or score > best["score"]:
                best = {"threshold": float(th), "metrics": m, "score": score}
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_batch_size = int(os.getenv("EVAL_BATCH_SIZE", str(config.BATCH_SIZE)))
    checkpoint_dir = Path(os.getenv("TUNE_CKPT_DIR", str(config.FEDERATED_CKPT_DIR)))
    val_ds, test_ds = build_splits()
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )

    ckpts = sorted(checkpoint_dir.glob("federated_model_round_*.pth"))
    if not ckpts:
        raise FileNotFoundError("Federated checkpoint bulunamadı.")

    best = None
    for ckpt in ckpts:
        round_no = int(ckpt.stem.split("_")[-1])
        checkpoint = torch.load(ckpt, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model = build_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict(state_dict)

        y_val, p_val = collect_probs(model, val_loader, device)
        pick = select_threshold(y_val, p_val)
        score = (
            pick["metrics"]["macro_f1"],
            pick["metrics"]["balanced_accuracy"],
            pick["metrics"]["malign_recall"],
        )
        print(
            f"round={round_no} threshold={pick['threshold']:.2f} "
            f"val_recall={pick['metrics']['malign_recall']:.4f} "
            f"val_macro_f1={pick['metrics']['macro_f1']:.4f} "
            f"val_bal_acc={pick['metrics']['balanced_accuracy']:.4f}"
        )

        if best is None or score > best["score"]:
            best = {
                "round": round_no,
                "checkpoint": str(ckpt.resolve().relative_to(ROOT_DIR)),
                "threshold": pick["threshold"],
                "val_metrics": pick["metrics"],
                "score": score,
                "model": model,
            }

    y_test, p_test = collect_probs(best["model"], test_loader, device)
    test_metrics = metrics_at_threshold(y_test, p_test, best["threshold"])

    out = {
        "recommended_checkpoint": best["checkpoint"],
        "recommended_round": best["round"],
        "recommended_threshold": best["threshold"],
        "selection_policy": "validation threshold sweep: maximize macro_f1/balanced_accuracy with recall and precision floors",
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
    }

    out_path = ROOT_DIR / "checkpoints" / "deployment_config.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
