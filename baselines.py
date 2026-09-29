"""
baselines.py — classical baselines on the SAME split and preprocessing as KG-HTT.

Uses the fixed train/test split saved by notebook 02 (the whole training split;
no validation set is needed), standardises on the training split, and trains
with balanced class weights (same as KG-HTT) unless --no_class_weight.

Example
    python baselines.py --data_dir data/prepared --dataset tcga
    python baselines.py --data_dir data/prepared --dataset gtex --models lr xgb
Outputs: results/baselines/<dataset>[_tag]/<model>/{run.json, predictions.npz}
"""
import argparse, json, os, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/prepared")
p.add_argument("--dataset", choices=["gtex", "tcga"], required=True)
p.add_argument("--models", nargs="+", default=["lr", "rf", "xgb"], choices=["lr", "rf", "xgb"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--out_dir", default="results/baselines")
p.add_argument("--n_jobs", type=int, default=4)
p.add_argument("--no_class_weight", action="store_true")
p.add_argument("--xgb_rounds", type=int, default=200)
p.add_argument("--limit_train", type=int, default=0, help="smoke test: use only N training samples")
p.add_argument("--model_path", default="kghtt_model_v3.py", help="only for fast_auc")
args = p.parse_args()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, log_loss, confusion_matrix
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(args.model_path)) or ".")
from kghtt_model_v3 import fast_auc

# ---------------- data (same split as KG-HTT) ----------------
d = np.load(os.path.join(args.data_dir, f"{args.dataset}.npz"), allow_pickle=True)
X, y, classes = d["X"], d["y"], list(d["classes"])
tr, te = d["train_idx"], d["test_idx"]
if args.limit_train:                                   # smoke test: stratified subset of the training split
    from sklearn.model_selection import train_test_split
    tr, _ = train_test_split(tr, train_size=args.limit_train, stratify=y[tr], random_state=0)
n_classes = len(classes)
scaler = StandardScaler().fit(X[tr])
Xtr, Xte = scaler.transform(X[tr]).astype(np.float32), scaler.transform(X[te]).astype(np.float32)
ytr, yte = y[tr], y[te]
counts = np.bincount(ytr, minlength=n_classes)
w_class = len(ytr) / (n_classes * np.maximum(counts, 1))
sample_weight = None if args.no_class_weight else w_class[ytr]
cw = None if args.no_class_weight else "balanced"
print(f"{args.dataset}: X {X.shape}, {n_classes} classes, train {len(tr)} / test {len(te)}, class_weight={'balanced' if cw else 'none'}")

def full_proba(model, Xte):
    """probabilities on all n_classes columns, even if a class is absent from the (sub)training set"""
    pr = model.predict_proba(Xte)
    if pr.shape[1] == n_classes:
        return pr
    full = np.zeros((len(Xte), n_classes), dtype=pr.dtype)
    full[:, np.asarray(model.classes_, dtype=int)] = pr
    return full

def evaluate(name, proba, fit_time):
    pred = proba.argmax(axis=1)
    m = {"balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
         "accuracy": float(accuracy_score(yte, pred)),
         "macro_f1": float(f1_score(yte, pred, average="macro")),
         "macro_auc_ovr": float(fast_auc(yte, proba)),
         "log_loss": float(log_loss(yte, proba, labels=list(range(n_classes)))),
         "per_class_f1": dict(zip(classes, f1_score(yte, pred, average=None).round(4).tolist())),
         "fit_time_min": round(fit_time / 60, 2)}
    out = os.path.join(args.out_dir, args.dataset + (f"_{args.tag}" if args.tag else ""), name)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "predictions.npz"), p_test=proba, y_test=yte, test_idx=te,
                        confusion=confusion_matrix(yte, pred))
    json.dump({"args": vars(args), "metrics": m, "classes": classes}, open(os.path.join(out, "run.json"), "w"), indent=1)
    print(f"  {name:4s} | bal.acc {m['balanced_accuracy']:.4f} | acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} "
          f"| AUC {m['macro_auc_ovr']:.4f} | logloss {m['log_loss']:.4f} | {m['fit_time_min']:.1f} min")
    return m

results = {}
if "lr" in args.models:
    # multinomial logistic regression, L2 (lbfgs). L1/saga was dropped: it did not converge in 165 min and gave no sparsity.
    t0 = time.time()
    lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, class_weight=cw)
    lr.fit(Xtr, ytr)
    results["lr"] = evaluate("lr", full_proba(lr, Xte), time.time() - t0)

if "rf" in args.models:
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight=cw, n_jobs=args.n_jobs, random_state=args.seed)
    rf.fit(Xtr, ytr)
    results["rf"] = evaluate("rf", full_proba(rf, Xte), time.time() - t0)

if "xgb" in args.models:
    import xgboost as xgb
    t0 = time.time()
    clf = xgb.XGBClassifier(n_estimators=args.xgb_rounds, max_depth=6, learning_rate=0.1, subsample=0.8,
                            colsample_bytree=0.5, tree_method="hist", objective="multi:softprob",
                            n_jobs=args.n_jobs, random_state=args.seed, verbosity=0)
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    results["xgb"] = evaluate("xgb", full_proba(clf, Xte), time.time() - t0)

summary_path = os.path.join(args.out_dir, args.dataset + (f"_{args.tag}" if args.tag else ""), "summary.json")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_class_f1"} for k, v in results.items()},
          open(summary_path, "w"), indent=1)
print("saved", summary_path)
