"""
summarize_runs.py — one table with every run (KG-HTT and baselines).

Walks results/ for run.json files, extracts configuration and metrics, and writes
    results/summary_runs.csv        one row per run
    results/summary_by_config.csv   runs grouped by configuration (seeds aggregated: n, mean, sd)
and prints both, per dataset. Smoke tests (tag containing "smoke" or --limit_train > 0)
are excluded unless --include_smoke.

Usage
    python summarize_runs.py                     # default: results/
    python summarize_runs.py --results_dir results --include_smoke
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results_dir", default="results")
p.add_argument("--include_smoke", action="store_true")
args = p.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(args.results_dir, "**", "run.json"), recursive=True)):
    run = json.load(open(path))
    a, m = run.get("args", {}), run.get("metrics", {})
    is_baseline = os.sep + "baselines" + os.sep in path
    smoke = "smoke" in (a.get("tag") or "") or bool(a.get("limit_train"))
    if smoke and not args.include_smoke:
        continue
    if is_baseline:
        model = os.path.basename(os.path.dirname(path)).upper()          # LR / RF / XGB
        cfg = dict(model=model, mode="-", gene_pool="-", pathway_pool="-", D="-")
        best_ep, n_ep, minutes = np.nan, np.nan, m.get("fit_time_min")
    else:
        # runs made before the pooling options existed have no pooling keys -> mean / mean;
        # runs made with kghtt_model_v4 used "pooling" for what is now "pathway_pooling"
        cfg = dict(model="KG-HTT",
                   mode=a.get("mode", "sum"),
                   gene_pool=a.get("gene_pooling", "mean"),
                   pathway_pool=a.get("pathway_pooling", a.get("pooling", "mean")),
                   D=a.get("embed_dim", 32))
        best_ep, n_ep, minutes = m.get("best_epoch"), m.get("epochs_trained"), m.get("train_time_min")
    rows.append({
        "dataset": a.get("dataset"),
        **cfg,
        "seed": a.get("seed", 0),
        "bal_acc": m.get("balanced_accuracy"),
        "macro_f1": m.get("macro_f1"),
        "log_loss": m.get("log_loss"),
        "auc": m.get("macro_auc_ovr"),
        "best_epoch": best_ep, "epochs": n_ep, "minutes": minutes,
        "data_dir": a.get("data_dir", run.get("data_dir")),
        "model_file": a.get("model_path", "-") if not is_baseline else "-",
        "smoke": smoke,
        "run_dir": os.path.relpath(os.path.dirname(path), args.results_dir),
    })

if not rows:
    raise SystemExit(f"no run.json found under {args.results_dir}/")

df = pd.DataFrame(rows)
# logical order: KG-HTT from the simplest configuration to the richest, then baselines
rank = {"model": {"KG-HTT": 0, "LR": 1, "XGB": 2, "RF": 3},
        "gene_pool": {"mean": 0, "attention": 1},
        "pathway_pool": {"mean": 0, "attention": 1, "multi_attention": 2},
        "mode": {"sum": 0, "concat": 1}}
for col, r in rank.items():
    df[f"_{col}"] = df[col].map(r).fillna(9)
df["_D"] = pd.to_numeric(df["D"], errors="coerce").fillna(0)
df = df.sort_values(["dataset", "_model", "_D", "_gene_pool", "_pathway_pool", "_mode", "seed"]) \
       .drop(columns=[c for c in df.columns if c.startswith("_")]).reset_index(drop=True)
df.to_csv(os.path.join(args.results_dir, "summary_runs.csv"), index=False)

# ---- grouped by configuration: seeds aggregated ----
keys = ["dataset", "model", "mode", "gene_pool", "pathway_pool", "D", "data_dir"]
g = df.groupby(keys, dropna=False, sort=False)
by_cfg = g.agg(n_seeds=("seed", "count"),
               bal_acc_mean=("bal_acc", "mean"), bal_acc_sd=("bal_acc", "std"),
               log_loss_mean=("log_loss", "mean"), minutes_mean=("minutes", "mean")).reset_index()
by_cfg.to_csv(os.path.join(args.results_dir, "summary_by_config.csv"), index=False)

# ---- print ----
pd.set_option("display.width", 200)
show = ["model", "mode", "gene_pool", "pathway_pool", "D", "seed", "bal_acc", "macro_f1",
        "log_loss", "best_epoch", "epochs", "minutes", "run_dir"]
fmt = {"bal_acc": "{:.4f}".format, "macro_f1": "{:.4f}".format, "log_loss": "{:.3f}".format,
       "bal_acc_mean": "{:.4f}".format, "bal_acc_sd": lambda v: "" if pd.isna(v) else f"{v:.4f}",
       "log_loss_mean": "{:.3f}".format, "minutes": lambda v: "" if pd.isna(v) else f"{v:.1f}",
       "minutes_mean": lambda v: "" if pd.isna(v) else f"{v:.1f}",
       "best_epoch": lambda v: "" if pd.isna(v) else f"{int(v)}", "epochs": lambda v: "" if pd.isna(v) else f"{int(v)}"}
for ds in df["dataset"].unique():
    print(f"\n{'=' * 30} {ds.upper()} — one row per run {'=' * 30}")
    print(df.loc[df["dataset"] == ds, show].to_string(index=False, formatters=fmt, na_rep=""))
    print(f"\n{'-' * 30} {ds.upper()} — by configuration (seeds aggregated) {'-' * 30}")
    cols = ["model", "mode", "gene_pool", "pathway_pool", "D", "n_seeds",
            "bal_acc_mean", "bal_acc_sd", "log_loss_mean", "minutes_mean"]
    print(by_cfg.loc[by_cfg["dataset"] == ds, cols].to_string(index=False, formatters=fmt, na_rep=""))

other_dirs = sorted(set(df["data_dir"].dropna()))
if len(other_dirs) > 1:
    print(f"\nNOTE: runs come from different data dirs {other_dirs} — only compare within the same one.")
print(f"\n{len(df)} runs -> {args.results_dir}/summary_runs.csv, {args.results_dir}/summary_by_config.csv")
