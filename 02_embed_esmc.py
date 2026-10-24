#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

_W = os.environ.get("ESMC_WEIGHTS")
if not _W:
    _hits = sorted(Path.home().glob(
        ".cache/huggingface/hub/models--EvolutionaryScale--esmc-300m-2024-12/"
        "snapshots/*/data/weights/esmc_300m_2024_12_v0.pth"))
    _W = str(_hits[0]) if _hits else ""
WEIGHTS = _W

MODEL_NOTE = "EvolutionaryScale/esmc-300m-2024-12 (local .pth, esm 3.2.1 API)"


def load_esmc(device: str = "cpu"):
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    if not WEIGHTS or not Path(WEIGHTS).exists():
        raise SystemExit(
            "ESMC weights not found. Expected the HuggingFace cache at\n"
            "  ~/.cache/huggingface/hub/models--EvolutionaryScale--"
            "esmc-300m-2024-12/snapshots/*/data/weights/esmc_300m_2024_12_v0.pth\n"
            "or set $ESMC_WEIGHTS to the .pth path."
        )
    t0 = time.time()
    with torch.device(device):
        model = ESMC(
            d_model=960, n_heads=15, n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    print(f"  loaded ESMC-300M in {time.time() - t0:.1f}s "
          f"from {Path(WEIGHTS).name}")
    return model


def main() -> None:
    meta = pd.read_csv(ROOT / "data" / "proteins.csv")
    print(f"{len(meta)} proteins from data/proteins.csv")

    torch.set_num_threads(8)
    print(f"Loading {MODEL_NOTE} on cpu")
    model = load_esmc("cpu")

    from esm.sdk.api import ESMProtein, LogitsConfig

    cfg = LogitsConfig(sequence=True, return_embeddings=True,
                       return_hidden_states=True)

    final, layers = [], []
    t0 = time.time()
    with torch.no_grad():
        for k, seq in enumerate(meta["sequence"].tolist()):
            out = model.logits(model.encode(ESMProtein(sequence=seq)), cfg)

            emb = out.embeddings[0]
            final.append(emb[1:-1].mean(0).float().numpy())

            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                h = hs[:, 0] if hs.dim() == 4 else hs
                layers.append(h[:, 1:-1, :].mean(1).float().numpy())

            if k % 10 == 0:
                print(f"  embedded {k}/{len(meta)}  L={len(seq)}  "
                      f"{time.time() - t0:.0f}s", flush=True)

    arr = np.stack(final)
    np.save(RESULTS / "esmc300m_meanpool.npy", arr)
    (RESULTS / "protein_ids.txt").write_text("\n".join(meta["id"].tolist()) + "\n")
    print(f"\nSaved {arr.shape} final-layer embeddings "
          f"-> results/esmc300m_meanpool.npy")

    if layers:
        lay = np.stack(layers)
        np.save(RESULTS / "esmc300m_layers.npy", lay)
        print(f"Saved {lay.shape} per-layer embeddings "
              f"-> results/esmc300m_layers.npy")
    else:
        print("! hidden_states unavailable; layer sweep will not be possible")

    print(f"model={MODEL_NOTE} device=cpu elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
