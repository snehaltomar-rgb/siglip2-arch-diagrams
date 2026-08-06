"""eval_v2.py — paradigm-v2 test-set eval driver.

Reads a manifest, encodes every row with:
  1. vanilla SigLIP2-L (baseline, pooler_output, no fine-tune) — always
  2. a given SSL-FT ckpt — only if --ckpt is supplied

Writes:
  eval_results.json            {baseline: {...}, ssl_ft: {...} | null, meta: {...}}
  eval_results.md              human-readable summary
  embeddings.npz  (optional)   Q_ssl, K_ssl, Q_base, K_base, pids, idxs

Callable as a script AND as `from eval_v2 import run_eval`, which the trainer
uses at each ckpt without spawning a subprocess.

Baseline is cached — if `baseline_cache.npz` exists in --out-dir, its
embeddings + metrics are reused (baseline is deterministic and doesn't move
between ckpts). Set --recompute-baseline to force a rerun.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssl_smoketest_siglip2 import (  # noqa: E402
    IMG_SIZE, MODEL_ID, AsymmetricSigLIP2, SmoketestDataset, collate_fn,
)

log = logging.getLogger("eval_v2")


# ---------------------------------------------------------------------------
# Baseline: vanilla SigLIP2 vision tower, pooler_output for both sides
# ---------------------------------------------------------------------------

class BaselineSigLIP2(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoModel
        full = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype)
        self.backbone = full.vision_model
        if hasattr(full, "text_model"):
            del full.text_model
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        vec = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state.mean(dim=1)
        return F.normalize(vec.float(), dim=-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(Q: torch.Tensor, K: torch.Tensor, pids: torch.Tensor,
                    ks=(1, 5, 10)) -> dict:
    N = Q.shape[0]
    sims = Q @ K.T
    topk_vals, topk_idx = sims.topk(max(ks), dim=1)
    tgt = torch.arange(N, device=Q.device).unsqueeze(1)
    hit_img = (topk_idx == tgt)
    metrics = {}
    for k in ks:
        metrics[f"image_r{k}"] = float(hit_img[:, :k].any(dim=1).float().mean().item())
    ranks = hit_img.float().argmax(dim=1) + 1
    any_hit = hit_img.any(dim=1)
    mrr = torch.zeros(N, device=Q.device)
    mrr[any_hit] = 1.0 / ranks[any_hit].float()
    metrics["image_mrr"] = float(mrr.mean().item())
    tgt_pid = pids.unsqueeze(1)
    retrieved_pid = pids[topk_idx]
    hit_prod = retrieved_pid == tgt_pid
    for k in ks:
        metrics[f"product_r{k}"] = float(hit_prod[:, :k].any(dim=1).float().mean().item())
    metrics["n_queries"] = int(N)
    return metrics


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode(model, loader, device, dtype, kind: str, log_every: int = 40):
    model.eval()
    Qs, Ks, Ps, Is = [], [], [], []
    t0 = time.time()
    n_seen = 0
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        wild = batch["wild"].to(device=device, dtype=dtype, non_blocking=True)
        cat = batch["catalog"].to(device=device, dtype=dtype, non_blocking=True)
        if kind == "ssl":
            q = model.encode_query(wild).float()
            k = model.encode_target(cat).float()
        else:
            q = model.encode(wild)
            k = model.encode(cat)
        Qs.append(q); Ks.append(k)
        Ps.append(batch["pid"].to(device))
        Is.append(batch["idx"].to(device))
        n_seen += q.shape[0]
        if (i + 1) % log_every == 0:
            log.info(f"  {kind} batch {i+1}/{len(loader)}  rows={n_seen}  elapsed={time.time()-t0:.1f}s")
    return (torch.cat(Qs, dim=0), torch.cat(Ks, dim=0),
            torch.cat(Ps, dim=0), torch.cat(Is, dim=0))


# ---------------------------------------------------------------------------
# Public API used by trainer
# ---------------------------------------------------------------------------

def load_ssl_ft_from_ckpt(ckpt_path: Path, dtype: torch.dtype,
                          layer_idx: int, device: torch.device,
                          variant: str | None = None):
    """Load an SSL-FT ckpt. If ckpt carries `variant_name` (paradigm-v2 layout)
    or one is passed, instantiates `VariantModel` — necessary for variants
    whose query head has extra submodules (e.g. `projector.` prefix on the
    predictor variants v2 and v7, or LoRA A/B params on the backbone for v4-v7).
    Falls back to `AsymmetricSigLIP2` for legacy ckpts that pre-date variants."""
    log.info(f"loading SSL-FT ckpt: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_variant = ckpt.get("variant_name") if isinstance(ckpt, dict) else None
    variant = variant or ckpt_variant
    # Delayed import: train_v2 already imports from this module, so a
    # top-level import would create a cycle.
    if variant and variant != "v0_baseline":
        from train_v2 import VariantModel  # noqa: E402
        model = VariantModel(variant_name=variant, layer_idx=layer_idx, dtype=dtype)
        log.info(f"  using VariantModel(variant={variant!r})")
    else:
        model = AsymmetricSigLIP2(layer_idx=layer_idx, dtype=dtype, unfreeze_last_n=0)
        log.info(f"  using AsymmetricSigLIP2 (variant={variant!r})")
    if "model" in ckpt:
        sd = ckpt["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        head_missing = [k for k in missing if k.startswith(("query_head.", "target_head."))]
        if head_missing:
            raise RuntimeError(f"missing head params in ckpt: {head_missing[:5]}...")
    elif "query_head" in ckpt:
        model.query_head.load_state_dict(ckpt["query_head"])
        model.target_head.load_state_dict(ckpt["target_head"])
    else:
        raise RuntimeError(f"unrecognised ckpt layout, keys={list(ckpt)}")
    model.to(device)
    model.query_head.to(dtype=dtype)
    model.target_head.to(dtype=dtype)
    return model


def run_eval(test_manifest: Path, cache_dir: Path, out_dir: Path,
             ckpt: Path | None = None,
             batch_size: int = 64, num_workers: int = 4,
             layer_idx: int = 22, dtype_str: str = "bf16",
             device: str | torch.device = "cuda",
             recompute_baseline: bool = False,
             save_embeddings: bool = False,
             log_prefix: str = "",
             baseline_cache_dir: Path | None = None) -> dict:
    """Encode + score. Baseline cached in `baseline_cache_dir` (default: out_dir).

    Passing a shared `baseline_cache_dir` lets many runs against the SAME test
    manifest reuse one baseline pass — the baseline embeddings depend only on
    (test manifest, cache_dir, model), so they're safe to share across runs
    with different ckpts. Cost: 1 baseline encode instead of N.
    """
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    device = torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SmoketestDataset(str(test_manifest), Path(cache_dir), size=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)
    n_expected = len(dataset)
    log.info(f"{log_prefix}dataset={test_manifest.name} rows={n_expected} batches={len(loader)}")

    # ---- Baseline: encode + score (cache reuse) ----
    baseline_dir = Path(baseline_cache_dir) if baseline_cache_dir is not None else out_dir
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache = baseline_dir / "baseline_cache.npz"
    baseline_metrics_path = baseline_dir / "baseline_metrics.json"
    if baseline_cache.exists() and baseline_metrics_path.exists() and not recompute_baseline:
        z = np.load(baseline_cache)
        Q_base = torch.from_numpy(z["Q_base"]).float().to(device)
        K_base = torch.from_numpy(z["K_base"]).float().to(device)
        pids   = torch.from_numpy(z["pids"]).long().to(device)
        idxs   = torch.from_numpy(z["idxs"]).long().to(device)
        m_base = json.loads(baseline_metrics_path.read_text())
        log.info(f"{log_prefix}baseline reloaded from cache ({baseline_dir}): product_r1={m_base['product_r1']:.4f}")
    else:
        log.info(f"{log_prefix}encoding baseline (vanilla SigLIP2)…")
        model_b = BaselineSigLIP2(dtype=dtype).to(device)
        t0 = time.time()
        Q_base, K_base, pids, idxs = _encode(model_b, loader, device, dtype, "baseline")
        log.info(f"{log_prefix}baseline encode done in {time.time()-t0:.1f}s  ({Q_base.shape[0]} rows)")
        del model_b
        torch.cuda.empty_cache()
        m_base = compute_metrics(Q_base, K_base, pids)
        n_encoded = int(Q_base.shape[0])
        n_missing = int(n_expected - n_encoded)
        m_base["n_expected"] = int(n_expected)
        m_base["n_encoded"] = n_encoded
        m_base["n_missing"] = n_missing
        m_base["missing_frac"] = float(n_missing / n_expected) if n_expected else 0.0
        if n_missing > 0:
            log.warning(
                f"{log_prefix}baseline SILENT-DROP: {n_missing}/{n_expected} rows failed to load "
                f"({100.0*n_missing/n_expected:.2f}%). corpus size effectively {n_encoded} "
                f"— cross-run R@1 comparability requires same n_encoded."
            )
        np.savez(baseline_cache,
                 Q_base=Q_base.cpu().numpy().astype("float32"),
                 K_base=K_base.cpu().numpy().astype("float32"),
                 pids=pids.cpu().numpy().astype("int64"),
                 idxs=idxs.cpu().numpy().astype("int64"))
        baseline_metrics_path.write_text(json.dumps(m_base, indent=2))
        log.info(f"{log_prefix}baseline cache written → {baseline_dir}  product_r1={m_base['product_r1']:.4f}")

    # ---- SSL-FT: encode + score (only if ckpt provided) ----
    m_ssl = None
    Q_ssl = K_ssl = None
    if ckpt is not None:
        model_s = load_ssl_ft_from_ckpt(ckpt, dtype, layer_idx, device)
        t0 = time.time()
        Q_ssl, K_ssl, pids_s, idxs_s = _encode(model_s, loader, device, dtype, "ssl")
        assert torch.equal(pids, pids_s) and torch.equal(idxs, idxs_s), \
            "pid/idx order diverged between baseline and ssl passes"
        log.info(f"{log_prefix}ssl encode done in {time.time()-t0:.1f}s")
        del model_s
        torch.cuda.empty_cache()
        m_ssl = compute_metrics(Q_ssl, K_ssl, pids)
        log.info(f"{log_prefix}ssl product_r1={m_ssl['product_r1']:.4f}  "
                 f"Δ={m_ssl['product_r1']-m_base['product_r1']:+.4f}")

    if save_embeddings and Q_ssl is not None:
        np.savez(out_dir / "embeddings.npz",
                 Q_ssl=Q_ssl.cpu().numpy().astype("float32"),
                 K_ssl=K_ssl.cpu().numpy().astype("float32"),
                 Q_base=Q_base.cpu().numpy().astype("float32"),
                 K_base=K_base.cpu().numpy().astype("float32"),
                 pids=pids.cpu().numpy().astype("int64"),
                 idxs=idxs.cpu().numpy().astype("int64"))

    results = {
        "test_manifest": str(test_manifest),
        "cache_dir": str(cache_dir),
        "ckpt": str(ckpt) if ckpt is not None else None,
        "model_id": MODEL_ID,
        "img_size": IMG_SIZE,
        "dtype": dtype_str,
        "layer_idx": layer_idx,
        "baseline": m_base,
        "ssl_ft": m_ssl,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    # Markdown summary
    def fmt(x): return f"{x:.4f}" if isinstance(x, float) else str(x)
    lines = [f"# Eval — {test_manifest.name}", "",
             f"N = {m_base['n_queries']}  ckpt = `{ckpt}`", ""]
    lines.append("| metric | baseline | ssl_ft | Δ (ssl − base) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for m in ["product_r1", "product_r5", "product_r10",
              "image_r1", "image_r5", "image_r10", "image_mrr"]:
        b = m_base[m]
        s = m_ssl[m] if m_ssl else float("nan")
        d = (s - b) if m_ssl else float("nan")
        lines.append(f"| {m} | {fmt(b)} | {fmt(s) if m_ssl else '—'} | {fmt(d) if m_ssl else '—'} |")
    (out_dir / "eval_results.md").write_text("\n".join(lines) + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-manifest", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--layer-idx", type=int, default=22)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--recompute-baseline", action="store_true")
    p.add_argument("--save-embeddings", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S")
    run_eval(args.test_manifest, args.cache_dir, args.out_dir,
             ckpt=args.ckpt, batch_size=args.batch_size, num_workers=args.num_workers,
             layer_idx=args.layer_idx, dtype_str=args.dtype,
             recompute_baseline=args.recompute_baseline,
             save_embeddings=args.save_embeddings)


if __name__ == "__main__":
    main()
