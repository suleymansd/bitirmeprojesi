# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Undergraduate thesis project: binary (Malign vs. Benign) skin cancer classification on HAM10000, comparing a centralized baseline against a **DCGAN-augmented Federated Learning** pipeline (5 simulated hospital clients, FedAvg via Flower). Detailed walkthrough lives in `PROJE_REHBERI.md` (Turkish); folder layout is summarized in `klasor_mimarisi.txt`.

Label mapping is defined once in `src/data_loader.py`:
- Malign (1): `mel`, `bcc`, `akiec`
- Benign (0): `nv`, `bkl`, `df`, `vasc`

## Environment Setup

Python 3.11 (dev env: 3.11.9). **Always use a venv** — never install into the system Python. The project has two requirements files:

- `requirements.txt` — full set, used by **GPU machines running actual training**. Includes `matplotlib` + `seaborn` for evaluation PNGs.
- `requirements-dev.txt` — minimal set for **code-only development** (no training). Drops `matplotlib`/`seaborn`. Use this if you're only editing code; reinstall the full set if you touch `evaluate*.py`.

```bash
# First time setup (macOS / CPU-only dev)
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt   # or requirements.txt for training machines
```

For GPU machines use the CUDA index URL from the header of `requirements.txt` (e.g. `cu128`) instead of `whl/cpu`. Every new shell requires `source venv/bin/activate`; VS Code does this automatically once the interpreter (`./venv/bin/python`) is selected.

Raw HAM10000 data is **not** in git — place it at `data/raw/HAM10000_metadata.csv` plus `data/raw/HAM10000_images_part_{1,2}/*.jpg`. Model checkpoints and `data/` are git-ignored.

## End-to-End Pipeline

Each stage assumes the previous outputs exist on disk. Run from repo root.

```bash
# 1. Baseline (ResNet50, imbalanced data, AMP, 10 epochs)
python src/train_baseline.py
python src/evaluate.py                  # classification report + CM PNG on 10% test split

# 2. DCGAN on Malign-only (128x128, TTUR, label smoothing, dropout)
python src/train_gan.py
python src/generate_synthetic.py        # writes 6450 synth_malign_*.jpg to data/synthetic/generated_malign/

# 3. Build 5-client IID splits (real train + synthetic, seed=42)
python src/prepare_federated_splits.py  # writes data/federated_splits/client_{1..5}.csv

# 4. Federated training (Flower, 5 rounds, FedAvg, full 5/5 participation required)
python src/server.py                    # terminal 1 — must start first
python src/client.py --client_id 1      # terminals 2..6
# ... through --client_id 5

# 5. Evaluate final global model on the same 10% test split
python src/evaluate_federated.py
```

On Windows, `run_federated.ps1` spawns the server and all 5 clients in separate PowerShell windows. On macOS/Linux, launch them manually in separate terminals.

There is no test suite, linter config, or build step — this is a research script project.

## Architecture Notes

**Shared primitives (`src/federated_utils.py`, `src/data_loader.py`):** The transform pipeline (`SquarePad` → `Resize(224,224)` → `ToTensor` → ImageNet `Normalize`) and the `ResNet50 + Linear(→2)` model **must stay identical** across `train_baseline.py`, `client.py`, and both evaluation scripts. `federated_utils.build_transforms()` / `build_model()` are the single source of truth for the federated side; any change there must be mirrored (or the baseline retrained) for the comparison to be valid. `SquarePad` is letterbox padding so 128×128 synthetic images and ~600×450 real images both feed the network without distortion.

**Federated flow (Flower 1.29):** `server.py` defines `SaveFedAvg`, a `FedAvg` subclass that — after each round's `aggregate_fit` — writes the merged state dict to `checkpoints/federated/federated_model_round_{N}.pth`. Round 5's checkpoint is the final model consumed by `evaluate_federated.py`. The server seeds initial parameters with `torch.manual_seed(42)` so all clients start identically; it waits for **full participation** (`min_*_clients=5`). Clients do 1 local epoch per round with AMP.

**Parameter I/O contract:** `get_parameters` / `set_parameters` in `federated_utils.py` convert between PyTorch `state_dict` and Flower's `List[np.ndarray]`. They rely on `state_dict` key order being stable — don't reorder layers or add/remove parameters without updating the server's `model_template` and regenerating the initial parameters.

**Data splits:** The 80/10/10 train/val/test split in `train_baseline.py` uses a fixed seed, and `prepare_federated_splits.py` uses the **same** split so that `evaluate_federated.py` tests the global model on held-out data the federated clients never saw. Synthetic Malign images are injected only into the federated train pool, never into test.

**CSV schema for federated clients:** `client_{i}.csv` must have columns `image_path` (absolute path) and `label` (0/1). `FedClientDataset` validates this on load.

## Conventions

- In-code comments and docstrings are in Turkish; keep new comments in Turkish for consistency with the rest of the repo.
- Scripts resolve paths via `ROOT_DIR = Path(__file__).resolve().parents[1]` — always run them from anywhere; don't hardcode cwd.
- `num_workers` is forced to 0 on Windows (`os.name == "nt"`) to avoid DataLoader multiprocessing issues; preserve that guard when editing loader code.
