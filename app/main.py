from __future__ import annotations

import io
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import config  # noqa: E402
from federated_utils import build_model, build_transforms  # noqa: E402


CLASS_NAMES = {0: "Benign", 1: "Malign"}


class InferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_threshold = 0.5
        self.model, self.checkpoint_path = self._load_model()
        self.transform = build_transforms()
        self.model.eval()

    def _load_deployment_config(self) -> dict:
        cfg_path = ROOT_DIR / "checkpoints" / "deployment_config.json"
        if not cfg_path.is_file():
            return {}
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _select_checkpoint(self) -> Path:
        deploy_cfg = self._load_deployment_config()
        recommended = deploy_cfg.get("recommended_checkpoint")
        if recommended:
            ckpt = Path(recommended)
            # Allow both absolute and repo-relative checkpoint paths.
            if not ckpt.is_absolute():
                ckpt = ROOT_DIR / ckpt
            if ckpt.is_file():
                self.default_threshold = float(
                    deploy_cfg.get("recommended_threshold", 0.5)
                )
                return ckpt

        mode = os.getenv("MODEL_MODE", "federated").strip().lower()
        if mode == "baseline":
            ckpt = config.BASELINE_CKPT_DIR / "baseline_best_model.pth"
            if not ckpt.is_file():
                raise FileNotFoundError(f"Baseline checkpoint bulunamadı: {ckpt}")
            return ckpt

        round_env = os.getenv("FEDERATED_ROUND")
        if round_env:
            ckpt = config.FEDERATED_CKPT_DIR / f"federated_model_round_{int(round_env)}.pth"
            if not ckpt.is_file():
                raise FileNotFoundError(f"Belirtilen federated checkpoint bulunamadı: {ckpt}")
            return ckpt

        candidates = sorted(config.FEDERATED_CKPT_DIR.glob("federated_model_round_*.pth"))
        if not candidates:
            raise FileNotFoundError("Federated checkpoint bulunamadı.")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_model(self) -> Tuple[torch.nn.Module, Path]:
        ckpt_path = self._select_checkpoint()
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Baseline checkpoint içinde optimizer vb. de olabilir, sadece state_dict alıyoruz.
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        model = build_model(num_classes=config.NUM_CLASSES, pretrained=False).to(self.device)
        model.load_state_dict(state_dict)
        return model, ckpt_path

    @torch.no_grad()
    def predict(self, image_bytes: bytes, threshold: float | None = None) -> Dict:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("Yüklenen dosya geçerli bir görsel değil.") from exc

        if threshold is None:
            threshold = self.default_threshold

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        benign = float(probs[0].item())
        malign = float(probs[1].item())
        pred = 1 if malign >= threshold else 0

        return {
            "prediction": {
                "label": CLASS_NAMES[pred],
                "class_id": pred,
                "risk_level": "high" if pred == 1 else "low",
            },
            "scores": {
                "benign": benign,
                "malign": malign,
            },
            "threshold": threshold,
            "model": {
                "checkpoint": str(self.checkpoint_path),
                "device": str(self.device),
            },
            "disclaimer": "Bu sistem klinik karar yerine geçmez; yalnızca araştırma/demo amaçlıdır.",
        }


app = FastAPI(title="Cilt Kanseri Tespit Demo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "web" / "static")), name="static")
service = InferenceService()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT_DIR / "web" / "templates" / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "checkpoint": str(service.checkpoint_path),
        "device": str(service.device),
    })


@app.get("/api/performance")
def performance() -> JSONResponse:
    deploy_cfg = service._load_deployment_config()
    return JSONResponse({
        "recommended_round": deploy_cfg.get("recommended_round"),
        "recommended_threshold": deploy_cfg.get("recommended_threshold"),
        "val_metrics": deploy_cfg.get("val_metrics", {}),
        "test_metrics": deploy_cfg.get("test_metrics", {}),
        "selection_policy": deploy_cfg.get("selection_policy", ""),
    })


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), threshold: float | None = None) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir görsel dosyası yükleyin.")

    if threshold is not None and not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="threshold 0.0 ile 1.0 arasında olmalı.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Boş dosya gönderildi.")

    try:
        result = service.predict(image_bytes, threshold=threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)
