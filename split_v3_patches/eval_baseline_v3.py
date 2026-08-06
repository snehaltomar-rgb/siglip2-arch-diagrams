"""eval_baseline_v3.py — vanilla SigLIP2-L image@R1 on split_v3 testset_unique.

Thin wrapper around scripts/eval_v2.run_eval that runs ONLY the baseline pass
(no --ckpt) against `split_v3/testset_unique.parquet` and writes a compact
summary at <out-dir>/split_v3_baseline_summary.json.

Note on the metric:
  testset_unique has 1 row per product_id, so the corpus (catalog side) has
  1 image per PID → image_r1 == product_r1 by construction. Both are reported
  from eval_v2's compute_metrics for cross-checking.

Usage (on workspace):
  python scripts/eval_baseline_v3.py \
      --testset-unique reports/training_manifest/split_v3/testset_unique.parquet \
      --cache-dir /home/ray/default/paradigm_v2/cache/split_v3 \
      --out-dir /home/ray/default/paradigm_v2/results/split_v3_baseline
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import run_eval  # noqa: E402


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset-unique", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="Image-tensor cache for SmoketestDataset. Fresh dir per manifest.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore any cached baseline_cache.npz and re-encode.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = run_eval(
        test_manifest=args.testset_unique,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        ckpt=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recompute_baseline=args.recompute,
        log_prefix="[split_v3_baseline] ",
    )

    baseline = results.get("baseline") if isinstance(results, dict) else None
    if baseline is None:
        baseline = json.loads((args.out_dir / "baseline_metrics.json").read_text())

    n_expected = int(baseline.get("n_expected") or baseline.get("n_queries") or 0) or None
    n_encoded = int(baseline.get("n_encoded") or baseline.get("n_queries") or 0) or None
    n_missing = int(baseline.get("n_missing") or 0)
    summary = {
        "split": "v3",
        "manifest": str(args.testset_unique),
        "n_expected": n_expected,
        "n_encoded": n_encoded,
        "n_missing": n_missing,
        "missing_frac": float(baseline.get("missing_frac") or 0.0),
        "model": "google/siglip2-large-patch16-512 (vanilla, pooler_output)",
        "image_r1": float(baseline["image_r1"]),
        "product_r1": float(baseline["product_r1"]),
        "image_r5": float(baseline["image_r5"]),
        "product_r5": float(baseline["product_r5"]),
        "image_mrr": float(baseline.get("image_mrr", -1.0)),
        "note": (
            "testset_unique is PID-unique -> image_r1 == product_r1 by construction. "
            "n_missing > 0 means some rows failed to download; R@1 is measured over "
            "n_encoded, not n_expected — cross-run comparability requires matching "
            "n_encoded (and matching image-cache bytes)."
        ),
    }
    summary_path = args.out_dir / "split_v3_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
