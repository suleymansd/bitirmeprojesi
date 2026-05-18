"""
Flower Federe Sunucusu  (FedAvg + Per-Round Checkpointing)
===========================================================

Bu sunucu, tüm istemcilerin her rauntta eğitime katılmasını bekler,
FedAvg stratejisiyle ağırlıkları birleştirir ve her raunt sonunda
birleşik global modeli `checkpoints/federated/federated_model_round_X.pth`
olarak diske kaydeder.

Toplam 20 raund çalışır. Son raundun çıktısı
`federated_model_round_20.pth`, `evaluate_federated.py` tarafından
orijinal test seti üzerinde değerlendirilecektir.

Kullanım:
---------
    python src/server.py

Ardından ayrı 2 terminalde:
    python src/client.py --client_id 1
    python src/client.py --client_id 2
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# src/ klasörünü import path'ine ekle
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config  # noqa: E402
from federated_utils import build_model, get_parameters, set_parameters  # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    "server_address": os.getenv("SERVER_ADDRESS", "0.0.0.0:8080"),
    "num_rounds": int(os.getenv("NUM_ROUNDS", str(config.NUM_ROUNDS))),
    "num_clients": config.NUM_CLIENTS,
    "checkpoint_dir": Path(os.getenv("FEDERATED_CKPT_DIR", str(config.FEDERATED_CKPT_DIR))),
    "ckpt_prefix": "federated_model_round",
    "num_classes": config.NUM_CLASSES,
    "seed": config.SPLIT_SEED,
}


# =============================================================================
# 2. ÖZEL STRATEJİ: FedAvg + her raunt sonunda checkpoint
# =============================================================================
class SaveFedAvg(FedAvg):
    """
    FedAvg'ın davranışını değiştirmeden, aggregate_fit'in hemen ardından
    birleştirilmiş global ağırlıkları diske (.pth) kaydeder.
    """

    def __init__(self, checkpoint_dir: Path, ckpt_prefix: str,
                 model_template: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_prefix = ckpt_prefix
        self.model_template = model_template
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Varsayılan FedAvg birleştirmesi
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Birleştirme başarılıysa PyTorch state_dict'e çevirip kaydet
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            set_parameters(self.model_template, ndarrays)

            out_path = (
                self.checkpoint_dir
                / f"{self.ckpt_prefix}_{server_round}.pth"
            )
            torch.save(
                {
                    "round": server_round,
                    "model_state_dict": self.model_template.state_dict(),
                    "num_clients_contributed": len(results),
                },
                out_path,
            )
            print(f"[Server] Round {server_round} global model kaydedildi: "
                  f"{out_path}")

        return aggregated_parameters, aggregated_metrics


# =============================================================================
# 3. YARDIMCI: ilk global parametreleri üret (deterministik başlangıç)
# =============================================================================
def get_initial_parameters() -> Parameters:
    """
    Tüm istemcilerin aynı başlangıç ağırlığıyla eğitime başlaması için
    sunucu tarafından üretilen ilk global model.
    """
    torch.manual_seed(CONFIG["seed"])
    # Network bağımlılığını kaldırmak için yerel başlangıç.
    model = build_model(num_classes=CONFIG["num_classes"], pretrained=False)
    return ndarrays_to_parameters(get_parameters(model))


# =============================================================================
# 4. AGREGASYON METRİKLERİ (weighted average)
# =============================================================================
def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]
                     ) -> Dict[str, Scalar]:
    """Örnek sayısına göre ağırlıklı ortalama metrik hesaplar."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    agg: Dict[str, float] = {}
    for n, m in metrics:
        for k, v in m.items():
            try:
                agg[k] = agg.get(k, 0.0) + float(v) * n
            except (TypeError, ValueError):
                continue
    return {k: v / total for k, v in agg.items()}


# =============================================================================
# 5. ANA AKIŞ
# =============================================================================
def main() -> None:
    cfg = CONFIG

    # Strateji: tüm istemcilerin (NUM_CLIENTS/NUM_CLIENTS) katılması şart
    strategy = SaveFedAvg(
        checkpoint_dir=cfg["checkpoint_dir"],
        ckpt_prefix=cfg["ckpt_prefix"],
        model_template=build_model(num_classes=cfg["num_classes"], pretrained=False),

        # FedAvg kwargs
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg["num_clients"],
        min_evaluate_clients=cfg["num_clients"],
        min_available_clients=cfg["num_clients"],
        initial_parameters=get_initial_parameters(),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"[Server] Başlatılıyor: {cfg['server_address']}")
    print(f"[Server] Rounds: {cfg['num_rounds']} | "
          f"Clients: {cfg['num_clients']} (tam katılım bekleniyor)")
    print(f"[Server] Checkpoint dizini: {cfg['checkpoint_dir']}")

    fl.server.start_server(
        server_address=cfg["server_address"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
