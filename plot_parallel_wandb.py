# To Generate an interactive Parallel Coordinates plot from sweep_out per-run JSONs.
#
# Usage:
#   python plot_parallel_wandb.py --indir /content/drive/MyDrive/deepcompress_ckpt/sweep_out_v1 --outdir /content/drive/MyDrive/deepcompress_ckpt/sweep_out_v1 --out both --wandb --wandb_project deep_compress
#
# This script:
#  - loads all *_result.json files in --indir
#  - extracts useful config & metrics (robust to missing fields)
#  - builds a pandas DataFrame and draws a Plotly parallel coordinates chart
#  - saves interactive HTML and (optionally) PNG
#  - can also log the Plotly figure to Weights & Biases if --wandb is provided

import os
import glob
import json
import argparse
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

def load_run_jsons(indir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(indir, "*_result.json")))
    runs = []
    for f in files:
        try:
            j = json.load(open(f))
            runs.append(j)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
    return runs

def build_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        cfg = r.get("config", {})
        # pull commonly expected fields with robust fallbacks
        row = {
            "run_name": r.get("name") or cfg.get("name") or "run",
            "sparsity": cfg.get("sparsity", cfg.get("sparsity", None)),
            "bits_conv": cfg.get("bits_conv", cfg.get("bits_map", {}).get("features") if isinstance(cfg.get("bits_map"), dict) else None),
            "bits_fc": cfg.get("bits_fc", None),
            "act_bits": cfg.get("act_bits", None),
            "acc_baseline": r.get("acc_baseline", None),
            "acc_compressed": r.get("acc_compressed", None),
            "final_MB_huffman": r.get("final_MB_huffman", r.get("final_MB_huffman", None)),
            "final_MB_raw": r.get("final_MB_raw", None),
            "ratio_model_huffman": r.get("ratio_model_huffman", None),
            "ratio_model_raw": r.get("ratio_model_raw", None),
            "ratio_weights_raw": r.get("ratio_weights_raw", None),
            "ratio_activations": r.get("ratio_activations", None),
            "empirical_huffman_bits": r.get("empirical_huffman_bits", None),
            "elapsed_sec": r.get("elapsed_sec", None)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def plot_parallel_coordinates(df: pd.DataFrame, out_html: str, out_png: str = None, title: str = "Compression sweep — Parallel Coordinates"):
    # Choose columns; ensure they exist
    dims = []
    # prefer a specific order; include only existing columns
    preferred = ["sparsity", "bits_conv", "bits_fc", "act_bits",
                 "acc_compressed", "final_MB_huffman", "final_MB_raw",
                 "ratio_model_huffman", "ratio_model_raw", "ratio_weights_raw", "ratio_activations"]
    for p in preferred:
        if p in df.columns:
            dims.append(p)
    if not dims:
        raise ValueError("No suitable dimensions found in dataframe. Columns present: " + ", ".join(df.columns))

    # Fill NaNs with a sentinel so plotly can render; keep original for saving
    plot_df = df.copy().fillna(-1)

    fig = px.parallel_coordinates(plot_df, dimensions=dims, color="acc_compressed",
                                  color_continuous_scale=px.colors.sequential.Viridis,
                                  labels={d: d for d in dims},
                                  title=title,
                                  height=600, width=1200)
    # Save interactive HTML
    fig.write_html(out_html)
    print(f"Saved interactive HTML to {out_html}")

    # Save PNG if requested (requires kaleido)
    if out_png:
        try:
            fig.write_image(out_png)
            print(f"Saved PNG to {out_png}")
        except Exception as e:
            print("Could not save PNG (kaleido may be missing):", e)

    return fig

def log_to_wandb(fig, project: str = "deep_compress", name: str = "parallel_coords_plot"):
    if not _WANDB_AVAILABLE:
        print("wandb not installed; skipping wandb upload.")
        return
    run = wandb.init(project=project, name=name, reinit=True)
    # log the plotly figure as an interactive artifact (WandB supports Plotly)
    try:
        wandb.log({"parallel_coordinates": wandb.plotly.plot(fig)})
        print("Logged parallel coordinates to WandB.")
    except Exception as e:
        # fallback: save HTML and upload
        tmp_html = "parallel_coords_tmp.html"
        fig.write_html(tmp_html)
        wandb.save(tmp_html)
        print("WandB logging fallback: saved HTML and uploaded as artifact.", e)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="sweep_out", help="Directory with per-run *_result.json files")
    parser.add_argument("--outdir", type=str, default="sweep_out", help="Output directory for saved plots")
    parser.add_argument("--out", choices=["html", "png", "both"], default="both", help="Save format")
    parser.add_argument("--wandb", action="store_true", help="Also log the figure to Weights & Biases (requires wandb login)")
    parser.add_argument("--wandb_project", type=str, default="deep_compress", help="WandB project name (if --wandb)")
    parser.add_argument("--title", type=str, default="Compression sweep — Parallel Coordinates", help="Plot title")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = load_run_jsons(args.indir)
    if not runs:
        print("No run JSONs found in", args.indir)
        return
    df = build_dataframe(runs)
    if df.empty:
        print("DataFrame is empty after loading runs.")
        return

    # Prepare output paths
    base = os.path.join(args.outdir, "parallel_coords")
    out_html = base + ".html"
    out_png = base + ".png" if args.out in ("png", "both") else None

    fig = plot_parallel_coordinates(df, out_html, out_png, title=args.title)

    if args.wandb:
        log_to_wandb(fig, project=args.wandb_project)

    print("Done. DataFrame summary:")
    print(df.describe(include="all"))

if __name__ == "__main__":
    main()
