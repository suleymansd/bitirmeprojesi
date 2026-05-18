from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config  # noqa: E402
from federated_utils import FedClientDataset, build_model, build_train_transforms  # noqa: E402
from client import WeightedFocalLoss  # noqa: E402


def make_loss(dataset: FedClientDataset, device: torch.device, mode: str) -> nn.Module:
    n_b, n_m = dataset.class_counts()
    total = n_b + n_m
    weights = torch.tensor(
        [total / (2 * n_b), total / (2 * n_m)],
        dtype=torch.float32,
        device=device,
    )
    if mode == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    return WeightedFocalLoss(weight=weights, gamma=1.5)


def train_client(
    global_state: dict[str, torch.Tensor],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> tuple[dict[str, torch.Tensor], float]:
    model = build_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(global_state)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    running_loss, seen = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, running_loss / max(seen, 1)


def fedavg(states: list[dict[str, torch.Tensor]], sizes: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(sizes))
    averaged = copy.deepcopy(states[0])
    for key in averaged:
        if averaged[key].is_floating_point():
            averaged[key] = sum(state[key] * (size / total) for state, size in zip(states, sizes))
        else:
            averaged[key] = states[0][key]
    return averaged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local FedAvg training without gRPC")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["focal", "ce"], default="focal")
    parser.add_argument("--out_dir", type=str, default="checkpoints/federated_local_full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.SPLIT_SEED)
    transform = build_train_transforms()
    loaders, criteria, sizes = [], [], []
    for client_id in range(1, config.NUM_CLIENTS + 1):
        ds = FedClientDataset(config.FEDERATED_SPLITS_DIR / f"client_{client_id}.csv", transform=transform)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        loaders.append(loader)
        criteria.append(make_loss(ds, device, args.loss))
        sizes.append(len(ds))
        n_b, n_m = ds.class_counts()
        print(f"[Client {client_id}] n={len(ds)} benign={n_b} malign={n_m}", flush=True)

    global_model = build_model(num_classes=config.NUM_CLASSES, pretrained=False)
    global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

    for round_no in range(1, args.rounds + 1):
        client_states, losses = [], []
        print(f"[Round {round_no}] started", flush=True)
        for idx, (loader, criterion) in enumerate(zip(loaders, criteria), start=1):
            state, loss = train_client(
                global_state,
                loader,
                criterion,
                device,
                args.lr,
                args.weight_decay,
            )
            client_states.append(state)
            losses.append(loss)
            print(f"[Round {round_no}] client={idx} loss={loss:.4f}", flush=True)

        global_state = fedavg(client_states, sizes)
        ckpt_path = out_dir / f"federated_model_round_{round_no}.pth"
        torch.save(
            {
                "round": round_no,
                "model_state_dict": global_state,
                "num_clients_contributed": len(sizes),
                "train_loss": sum(loss * size for loss, size in zip(losses, sizes)) / sum(sizes),
            },
            ckpt_path,
        )
        print(f"[Round {round_no}] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
