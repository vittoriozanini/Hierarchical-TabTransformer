"""
kghtt_analysis.py — utilities for the analysis notebook (05_analysis.ipynb).

Loads trained runs, recomputes test predictions, and computes permutation
importance at the pathway and gene level. The model file itself is untouched.

Primary metric: macro one-vs-rest AUC drop (as in the original work).
Secondary metric: balanced log-loss increase, computed in the same pass.
Per-class AUC drops are always kept, so the per-class deep dive (B4) costs nothing extra.
"""
import json, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import rankdata


# ======================================================
# LOADING
# ======================================================

def load_run(run_dir, model_module, data_root="."):
    """
    Rebuild a trained model from its run directory.
    Returns dict with model, args, metrics, classes, test arrays and saved predictions.
    """
    import joblib
    run = json.load(open(os.path.join(run_dir, "run.json")))
    a, classes = run["args"], run["classes"]
    d = np.load(os.path.join(data_root, a["data_dir"], f"{a['dataset']}.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(data_root, a["data_dir"], "meta.json")))
    pre = joblib.load(os.path.join(run_dir, "preprocessors.joblib"))
    X, y, test_idx = d["X"], d["y"], d["test_idx"]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xc = pre["scaler"].transform(X[test_idx]).astype(np.float32)
        Xb = np.clip(pre["kbd"].transform(X[test_idx]), 0, a["n_bins"] - 1).astype(np.int32)

    model = model_module.HierarchicalTabTransformer(
        meta["pathway_map"], X.shape[1], n_bins=a["n_bins"], embed_dim=a["embed_dim"],
        num_heads=a["num_heads"], ff_dim=a["ff_dim"], local_layers=a["local_layers"],
        global_layers=a["global_layers"], dropout=a["dropout"], task="multiclass",
        n_classes=len(classes), mode=a["mode"], chunk_size=a["chunk_size"],
        gene_pooling=a.get("gene_pooling", "mean"),
        pathway_pooling=a.get("pathway_pooling", a.get("pooling", "mean")),
        n_queries=a.get("n_queries", 4))
    model([Xc[:2], Xb[:2]], training=False)
    model.load_weights(os.path.join(run_dir, "model.weights.h5"))

    saved = np.load(os.path.join(run_dir, "predictions.npz"))
    return dict(model=model, args=a, metrics=run["metrics"], classes=classes, meta=meta,
                X_cont=Xc, X_bins=Xb, y=y[test_idx], test_idx=test_idx,
                p_saved=saved["p_test"], run_dir=run_dir,
                name=f"{a['dataset']}_s{a['seed']}")


def check_reload(run, batch_size=64, tol=1e-4):
    """Recomputed test predictions must match the ones saved at training time."""
    p = run["model"].predict([run["X_cont"], run["X_bins"]], batch_size=batch_size, verbose=0)
    err = float(np.abs(p - run["p_saved"]).max())
    return err < tol, err, p


# ======================================================
# METRICS
# ======================================================

def auc_per_class(y, p):
    """
    One-vs-rest AUC for every class: (C,) array, NaN for classes absent from y.
    Rank-based, matches sklearn's roc_auc_score per class.
    """
    n, C = p.shape
    ranks = rankdata(p, axis=0)
    out = np.full(C, np.nan)
    for c in range(C):
        pos = (y == c)
        n_pos = int(pos.sum()); n_neg = n - n_pos
        if n_pos and n_neg:
            out[c] = (ranks[pos, c].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def logloss_per_class(y, p, eps=1e-12):
    """Mean log-loss within each class: (C,) array. The balanced log-loss is its mean."""
    ll = -np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))
    return np.array([ll[y == c].mean() if (y == c).any() else np.nan for c in range(p.shape[1])])


def _batched_predict(model, z, batch_size):
    return np.concatenate([model.predict_from_pathway_vectors(tf.constant(z[s:s + batch_size])).numpy()
                           for s in range(0, len(z), batch_size)], axis=0)


# ======================================================
# PATHWAY-LEVEL PERMUTATION IMPORTANCE
# ======================================================

def pathway_importance(run, n_perm=10, seed=0, batch_size=64, eval_idx=None,
                       pathways=None, store_preds_for=None, verbose=True, log_every=20):
    """
    For every pathway P: permute its genes jointly along the sample axis, inside P only,
    and re-run global blocks + head on the cached pathway vectors (exact shortcut:
    a joint permutation commutes with the per-sample local encoding).

    Returns (df, per_class, baseline, preds) where
      df        : DataFrame indexed by pathway with auc_drop, auc_drop_se, logloss_rise, n_genes
      per_class : (K, C) array of one-vs-rest AUC drops per class  -> used by the B4 deep dive
      baseline  : dict with auc_macro, auc_class, logloss_balanced, p0, eval_idx
      preds     : {pathway: (n_perm, n, C)} only for pathways in store_preds_for (for the CI)
    """
    model, y_all = run["model"], run["y"]
    idx = np.arange(len(y_all)) if eval_idx is None else np.asarray(eval_idx)
    Xc, Xb, y = run["X_cont"][idx], run["X_bins"][idx], y_all[idx]
    n = len(y)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    z = np.concatenate([model.compute_pathway_vectors(Xc[s:s + batch_size], Xb[s:s + batch_size]).numpy()
                        for s in range(0, n, batch_size)], axis=0)             # (n, K, D)
    p0 = _batched_predict(model, z, batch_size)
    a0, l0 = auc_per_class(y, p0), logloss_per_class(y, p0)
    if verbose:
        print(f"  cached z {z.shape} + baseline in {time.time() - t0:.0f}s | "
              f"macro AUC {np.nanmean(a0):.4f} | balanced log-loss {np.nanmean(l0):.4f}")

    perms = [rng.permutation(n) for _ in range(n_perm)]        # same permutations for every pathway
    names = pathways if pathways is not None else model.pathway_names
    store = set(store_preds_for or [])
    rows, per_class, preds_out = [], [], {}
    t1 = time.time()
    for i, name in enumerate(names):
        t = model.pathway_names.index(name)
        orig = z[:, t, :].copy()                                # modify one column only, then restore
        aucs, lls, keep = [], [], []
        for perm in perms:
            z[:, t, :] = orig[perm]
            pk = _batched_predict(model, z, batch_size)
            aucs.append(auc_per_class(y, pk))
            lls.append(logloss_per_class(y, pk))
            if name in store:
                keep.append(pk)
        z[:, t, :] = orig
        aucs, lls = np.array(aucs), np.array(lls)               # (n_perm, C)
        macro = np.nanmean(aucs, axis=1)
        rows.append({"pathway": name, "n_genes": len(model.pathway_map[name]),
                     "auc_drop": float(np.nanmean(a0) - macro.mean()),
                     "auc_drop_se": float(macro.std(ddof=1) / np.sqrt(n_perm)) if n_perm > 1 else np.nan,
                     "logloss_rise": float(np.nanmean(lls, axis=1).mean() - np.nanmean(l0))})
        per_class.append(np.nanmean(a0[None, :] - aucs, axis=0))
        if name in store:
            preds_out[name] = np.stack(keep)
        if verbose and (i + 1) % log_every == 0:
            el = time.time() - t1
            print(f"  [{i + 1}/{len(names)}] {el / 60:.1f} min elapsed, "
                  f"{el / (i + 1) * (len(names) - i - 1) / 60:.1f} min left")

    df = pd.DataFrame(rows).set_index("pathway")
    baseline = dict(auc_macro=float(np.nanmean(a0)), auc_class=a0,
                    logloss_balanced=float(np.nanmean(l0)), p0=p0, eval_idx=idx)
    return df, np.array(per_class), baseline, preds_out


def bootstrap_ci(y, p0, preds, n_boot=200, seed=0, use_perms=5, alpha=0.05):
    """
    Paired percentile bootstrap of the macro-AUC drop, computed on stored predictions
    (no extra forward pass). Resamples the test samples, using the same indices for the
    baseline and the permuted predictions, so the interval is on the paired difference.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    preds = preds[:use_perms]
    diffs = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        a0 = np.nanmean(auc_per_class(y[b], p0[b]))
        ak = np.mean([np.nanmean(auc_per_class(y[b], pk[b])) for pk in preds])
        diffs.append(a0 - ak)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ======================================================
# GENE-LEVEL: BETA WEIGHTS AND PERMUTATION IMPORTANCE
# ======================================================

def gene_pooling_weights(run, batch_size=64, eval_idx=None, per_class=False):
    """
    beta weights from a forward pass. Requires gene_pooling="attention".

    Returns (mean_beta, entropy[, by_class]):
      mean_beta : {pathway: (n_k,)}  beta averaged over samples
      entropy   : normalised entropy of beta per pathway, averaged over samples
                  (1 = uniform = the plain mean it replaces, 0 = all weight on one gene)
      by_class  : {pathway: (C, n_k)} beta averaged within each class, if per_class=True

    Only these summaries are kept: the full (n, n_k) weights are accumulated one batch at a
    time and discarded, because storing them for every pathway costs hundreds of MB per model.
    """
    model = run["model"]
    idx = np.arange(len(run["y"])) if eval_idx is None else np.asarray(eval_idx)
    Xc, Xb, y = run["X_cont"][idx], run["X_bins"][idx], run["y"][idx]
    C = len(run["classes"])
    tot, ent_sum, cls_sum, cls_n = {}, {}, {}, np.bincount(y, minlength=C)
    for s in range(0, len(idx), batch_size):
        _, _, _, _, beta = model.forward_with_attention([Xc[s:s + batch_size], Xb[s:s + batch_size]])
        yb = y[s:s + batch_size]
        for name, b in beta.items():
            b = np.asarray(b)
            n_k = b.shape[1]
            h = (-(b * np.log(np.clip(b, 1e-12, 1))).sum(axis=1) / np.log(n_k)) if n_k > 1 else np.ones(len(b))
            tot[name] = tot.get(name, 0) + b.sum(axis=0)
            ent_sum[name] = ent_sum.get(name, 0.0) + float(h.sum())
            if per_class:
                acc = cls_sum.setdefault(name, np.zeros((C, n_k)))
                np.add.at(acc, yb, b)
    n = len(idx)
    mean_beta = {k: v / n for k, v in tot.items()}
    ent = pd.Series({k: v / n for k, v in ent_sum.items()}, name="beta_entropy")
    if per_class:
        by_class = {k: v / np.maximum(cls_n, 1)[:, None] for k, v in cls_sum.items()}
        return mean_beta, ent, by_class
    return mean_beta, ent


def gene_importance(run, pathway, n_perm=10, seed=0, batch_size=64, eval_idx=None, verbose=True):
    """
    Permutation importance of each gene INSIDE one pathway: the gene is permuted only in
    this pathway, its copies in other pathways are untouched. Re-encoding is required here
    (a single-gene permutation mixes samples within the pathway), so this is the expensive
    branch — run it on a few pathways only.
    """
    model, y_all = run["model"], run["y"]
    idx = np.arange(len(y_all)) if eval_idx is None else np.asarray(eval_idx)
    Xc, Xb, y = run["X_cont"][idx], run["X_bins"][idx], y_all[idx]
    n = len(y)
    rng = np.random.default_rng(seed)
    t = model.pathway_names.index(pathway)
    genes = list(model.pathway_map[pathway])
    symbol = {g["col"]: (g["symbol"] or g["ensembl"]) for g in run["meta"]["genes"]}

    z = np.concatenate([model.compute_pathway_vectors(Xc[s:s + batch_size], Xb[s:s + batch_size]).numpy()
                        for s in range(0, n, batch_size)], axis=0)
    a0 = np.nanmean(auc_per_class(y, _batched_predict(model, z, batch_size)))
    perms = [rng.permutation(n) for _ in range(n_perm)]
    orig = z[:, t, :].copy()
    rows = []
    for j, gcol in enumerate(genes):
        aucs = []
        for perm in perms:
            # permute this one gene across samples in the whole input, then re-encode only
            # this pathway: its copies inside the other pathways keep the original values,
            # because every other pathway vector is read from the cache z.
            Xc_p, Xb_p = Xc.copy(), Xb.copy()
            Xc_p[:, gcol], Xb_p[:, gcol] = Xc[perm, gcol], Xb[perm, gcol]
            z[:, t, :] = np.concatenate([model.encode_pathway_subset(
                Xc_p[s:s + batch_size], Xb_p[s:s + batch_size], t).numpy()
                for s in range(0, n, batch_size)], axis=0)
            aucs.append(np.nanmean(auc_per_class(y, _batched_predict(model, z, batch_size))))
        z[:, t, :] = orig
        rows.append({"gene_col": gcol, "symbol": symbol.get(gcol, str(gcol)),
                     "auc_drop": float(a0 - np.mean(aucs)),
                     "auc_drop_se": float(np.std(aucs, ddof=1) / np.sqrt(n_perm)) if n_perm > 1 else np.nan})
        if verbose and (j + 1) % 25 == 0:
            print(f"    {j + 1}/{len(genes)} genes")
    return pd.DataFrame(rows).set_index("symbol").sort_values("auc_drop", ascending=False)


# ======================================================
# SELF-ATTENTION WEIGHTS
# ======================================================

def attention_summary(run, batch_size=32):
    """
    How uniform is the self-attention? One forward pass on the test set.
    Normalised entropy of each attention row: 1 = uniform over the keys, 0 = all the weight on one key.
    Returns global entropy and largest weight per sample, the attention received by each pathway
    (overall and per class), and the local entropy of each pathway (averaged over samples).
    """
    model, Xc, Xb, y = run["model"], run["X_cont"], run["X_bins"], run["y"]
    names = model.pathway_names
    K, C = len(names), len(run["classes"])
    recv_sum, recv_cls = np.zeros(K), np.zeros((C, K))
    g_ent, g_max = [], []
    loc_ent = np.zeros(K)
    n = len(y)
    for s in range(0, n, batch_size):
        _, local, glob, _, _ = model.forward_with_attention([Xc[s:s + batch_size], Xb[s:s + batch_size]])
        G = np.asarray(glob[-1])                                         # (b, heads, K, K)
        ent = -(G * np.log(np.clip(G, 1e-12, 1))).sum(-1) / np.log(K)
        g_ent.append(ent.mean(axis=(1, 2)))
        g_max.append(G.max(-1).mean(axis=(1, 2)))
        recv = G.mean(axis=1).mean(axis=1)                               # (b, K)
        recv_sum += recv.sum(0)
        np.add.at(recv_cls, y[s:s + batch_size], recv)
        for j, p in enumerate(names):
            L = np.asarray(local[p][-1])                                 # (b, heads, n_k, n_k)
            nk = L.shape[-1]
            if nk > 1:
                loc_ent[j] += float((-(L * np.log(np.clip(L, 1e-12, 1))).sum(-1) / np.log(nk)).mean(axis=(1, 2)).sum())
            else:
                loc_ent[j] += len(L)
    return dict(global_entropy=np.concatenate(g_ent), global_max=np.concatenate(g_max),
                received=pd.Series(recv_sum / n, index=names),
                received_by_class=pd.DataFrame(recv_cls / np.maximum(np.bincount(y, minlength=C), 1)[:, None],
                                               index=run["classes"], columns=names),
                local_entropy=pd.Series(loc_ent / n, index=names))
