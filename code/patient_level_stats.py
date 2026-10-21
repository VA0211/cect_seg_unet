"""Patient-level evaluation of test_results.csv files (one per model).

Usage:
    python patient_level_stats.py --csv CBAM=path/to/test_results.csv --csv Baseline=path/to/test_results.csv \
        --baseline Baseline [--test_dir /kaggle/input/cect-npy-full/npy_dataset/images] [--out summary.csv]

CSV files written by the patched test_2D_fully.py contain file/patient_id/phase columns. Older CSVs (metrics only)
are mapped to patients with --test_dir: rows follow the sorted list of test_*.npy files (shuffle=False).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["Dice", "IoU", "HD95", "Accuracy", "Precision", "Sensitivity", "Specificity"]
DIAGONAL = float(np.hypot(256, 256))  # HD95 assigned to empty predictions in the sensitivity analysis


def load(path, test_dir=None, mask_dir=None):
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        if not test_dir:
            raise ValueError(f"{path} has no patient_id column; pass --test_dir")
        files = sorted(f for f in os.listdir(test_dir) if f.startswith("test"))
        if len(files) != len(df):
            raise ValueError(f"{path}: {len(df)} rows but {len(files)} test files")
        df.insert(0, "file", files)
        df.insert(1, "patient_id", [f.split("_")[1] for f in files])
        df.insert(2, "phase", [f.split("_")[2] for f in files])
    if "empty_pred" not in df.columns:
        # old CSVs: empty prediction <=> HD95 stored as 999 (ground truth is never empty in the test set)
        df["empty_pred"] = (df["HD95"] >= 999).astype(int)
        df.loc[df["empty_pred"] == 1, "HD95"] = np.nan
    if mask_dir:
        df = fix_empty_metrics(df, mask_dir)
    df["HD95_sens"] = df["HD95"].fillna(DIAGONAL)
    return df


def fix_empty_metrics(df, mask_dir):
    """Older test_2D_fully.py versions set Accuracy/Precision/Sensitivity/Specificity to 0 when the prediction
    was empty. With TP = FP = 0 and FN = |G|: Accuracy = 1 - |G|/N, Specificity = 1, Sensitivity = 0 and
    Precision is undefined (NaN)."""
    rows = df.index[df["empty_pred"] == 1]
    for i in rows:
        g = np.load(os.path.join(mask_dir, df.at[i, "file"]))
        n, g_sum = g.size, int((g > 0).sum())
        df.at[i, "Accuracy"] = (n - g_sum) / n
        df.at[i, "Specificity"] = 1.0
        df.at[i, "Sensitivity"] = 0.0
        df.at[i, "Precision"] = np.nan
    return df


def per_patient(df):
    g = df.groupby("patient_id")
    out = g[METRICS + ["HD95_sens"]].mean()
    out["empty_rate"] = g["empty_pred"].mean()
    return out


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="NAME=path/to/test_results.csv")
    ap.add_argument("--baseline", help="model name used as reference for paired tests")
    ap.add_argument("--test_dir", help="npy images folder, needed for CSVs without patient_id")
    ap.add_argument("--mask_dir", help="npy masks folder: recompute pixel-wise metrics of empty predictions")
    ap.add_argument("--slice_out", help="optional folder to write the corrected per-slice CSVs")
    ap.add_argument("--out", default="patient_level_summary.csv")
    args = ap.parse_args()

    pp = {}
    rows = []
    for item in args.csv:
        name, path = item.split("=", 1)
        df = load(path, args.test_dir, args.mask_dir)
        if args.slice_out:
            os.makedirs(args.slice_out, exist_ok=True)
            df.to_csv(os.path.join(args.slice_out, f"test_results_{name}.csv"), index=False)
        p = per_patient(df)
        pp[name] = p
        row = {"model": name, "patients": len(p), "slices": len(df),
               **{f"{m}_slice_mean": df[m].mean() for m in METRICS},
               "empty_pred_n": int(df["empty_pred"].sum()), "empty_pred_pct": 100 * df["empty_pred"].mean()}
        for m in METRICS + ["HD95_sens"]:
            vals = p[m].dropna()
            lo, hi = boot_ci(vals)
            row[f"{m}_mean"], row[f"{m}_sd"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = vals.mean(), vals.std(), lo, hi
        rows.append(row)
    summary = pd.DataFrame(rows)

    if args.baseline:
        base = pp[args.baseline]
        names = [n for n in pp if n != args.baseline]
        raw = []
        for n in names:
            a, b = base["Dice"].align(pp[n]["Dice"], join="inner")
            raw.append(wilcoxon(b, a).pvalue)
        adj = holm(np.array(raw)) if raw else []
        summary["p_dice_vs_baseline"] = summary["model"].map(dict(zip(names, raw)))
        summary["p_dice_vs_baseline_holm"] = summary["model"].map(dict(zip(names, adj)))

    summary.to_csv(args.out, index=False)
    cols = ["model", "patients", "empty_pred_pct", "Dice_mean", "Dice_ci_lo", "Dice_ci_hi", "IoU_mean",
            "HD95_mean", "HD95_sens_mean"] + [c for c in summary.columns if c.startswith("p_")]
    print(summary[cols].round(4).to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
