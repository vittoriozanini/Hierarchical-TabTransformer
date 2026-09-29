"""
train_kghtt.py — train and evaluate KG-HTT v3 on a prepared dataset (GTEx or TCGA).

Reads data/prepared*/{gtex,tcga}.npz + meta.json produced by notebook 02 (fixed
train/test split), carves a stratified validation set out of the training split,
fits scaler + quantile binning on the training part only, trains with early
stopping on the validation loss, evaluates on the test split in batches, and saves
everything the analysis script needs (weights, preprocessors, predictions).

Example
    python train_kghtt.py --data_dir data/prepared --dataset tcga --mode sum --seed 0
    python train_kghtt.py --data_dir data/prepared --dataset gtex --mode concat --seed 1 --tag pilot
Smoke test (2-3 min):
    python train_kghtt.py --data_dir data/prepared --dataset tcga --limit_train 400 --epochs 1 --tag smoke
"""
import argparse, json, os, sys, time, resource
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/prepared")
p.add_argument("--dataset", choices=["gtex", "tcga"], required=True)
p.add_argument("--mode", choices=["sum", "concat"], default="sum")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--out_dir", default="results")
# architecture
p.add_argument("--embed_dim", type=int, default=32)
p.add_argument("--num_heads", type=int, default=2)
p.add_argument("--ff_dim", type=int, default=64)
p.add_argument("--local_layers", type=int, default=1)
p.add_argument("--global_layers", type=int, default=1)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--n_bins", type=int, default=6)
p.add_argument("--chunk_size", type=int, default=16)
p.add_argument("--gene_pooling", choices=["mean", "attention"], default="mean",
               help="genes -> one vector per pathway: masked mean, or attention-weighted (beta)")
p.add_argument("--pathway_pooling", choices=["mean", "attention", "multi_attention"], default="mean",
               help="pathways -> head input: mean, or attention-weighted (alpha), with 1 or n_queries queries")
p.add_argument("--n_queries", type=int, default=4, help="only for --pathway_pooling multi_attention")
# training
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--patience", type=int, default=4)
p.add_argument("--min_delta", type=float, default=0.005,
               help="minimum val_loss improvement counted by early stopping (0.005 stops the "
                    "epochs that only shave noise off the plateau)")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight_decay", type=float, default=1e-4)
p.add_argument("--val_frac", type=float, default=0.15, help="fraction of the training split used for validation")
p.add_argument("--no_class_weight", action="store_true")
p.add_argument("--limit_train", type=int, default=0, help="use only N training samples (smoke test)")
p.add_argument("--model_path", default="kghtt_model_v5.py", help="path to the model file")
args = p.parse_args()

# ---------------- reproducibility ----------------
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
import tensorflow as tf
tf.random.set_seed(args.seed)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, log_loss, confusion_matrix
import joblib
# import the model module named by --model_path (kghtt_model_v3.py, kghtt_model_v4.py, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(args.model_path)) or ".")
_model_module = __import__(os.path.splitext(os.path.basename(args.model_path))[0])
HierarchicalTabTransformer, fast_auc = _model_module.HierarchicalTabTransformer, _model_module.fast_auc

pool_tag = ("" if args.gene_pooling == "mean" else "_gene-attn") + \
           ("" if args.pathway_pooling == "mean" else f"_pw-{args.pathway_pooling}")
run_name = f"{args.dataset}_{args.mode}{pool_tag}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
out = os.path.join(args.out_dir, run_name)
os.makedirs(out, exist_ok=True)
print(f"run: {run_name}\nTF {tf.__version__}, threads: {tf.config.threading.get_intra_op_parallelism_threads() or 'auto'}")

# ---------------- data ----------------
d = np.load(os.path.join(args.data_dir, f"{args.dataset}.npz"), allow_pickle=True)
meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
X, y, classes = d["X"], d["y"], list(d["classes"])
train_idx, test_idx = d["train_idx"], d["test_idx"]
pathway_map = meta["pathway_map"]
n_classes, n_features = len(classes), X.shape[1]
n_max = max(len(v) for v in pathway_map.values())
print(f"X {X.shape}, {n_classes} classes, {len(pathway_map)} pathways, n_max {n_max}, "
      f"train {len(train_idx)} / test {len(test_idx)}")

# validation split carved out of the fixed training split (stratified, seeded)
tr_idx, va_idx = train_test_split(train_idx, test_size=args.val_frac, stratify=y[train_idx], random_state=args.seed)
if args.limit_train:
    tr_idx = tr_idx[:args.limit_train]
    va_idx = va_idx[:max(50, args.limit_train // 5)]
print(f"train {len(tr_idx)} / val {len(va_idx)} / test {len(test_idx)}")

# preprocessing fitted on the training part only
scaler = StandardScaler().fit(X[tr_idx])
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kbd = KBinsDiscretizer(n_bins=args.n_bins, encode="ordinal", strategy="quantile").fit(X[tr_idx])

def prep(idx):
    xc = scaler.transform(X[idx]).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xb = np.clip(kbd.transform(X[idx]), 0, args.n_bins - 1).astype(np.int32)
    return xc, xb

Xc_tr, Xb_tr = prep(tr_idx); Xc_va, Xb_va = prep(va_idx); Xc_te, Xb_te = prep(test_idx)
y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[test_idx]
eff_bins = np.array([len(e) - 1 for e in kbd.bin_edges_])
print(f"genes with all {args.n_bins} bins: {(eff_bins == args.n_bins).mean():.1%}")

class_weight = None
if not args.no_class_weight:
    counts = np.bincount(y_tr, minlength=n_classes)
    w = len(y_tr) / (n_classes * np.maximum(counts, 1))
    class_weight = {i: float(w[i]) for i in range(n_classes)}

# ---------------- model ----------------
model = HierarchicalTabTransformer(
    pathway_map, n_features, n_bins=args.n_bins, embed_dim=args.embed_dim, num_heads=args.num_heads,
    ff_dim=args.ff_dim, local_layers=args.local_layers, global_layers=args.global_layers,
    dropout=args.dropout, task="multiclass", n_classes=n_classes, mode=args.mode, chunk_size=args.chunk_size,
    gene_pooling=args.gene_pooling, pathway_pooling=args.pathway_pooling, n_queries=args.n_queries)
model([Xc_tr[:2], Xb_tr[:2]], training=False)   # build
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
n_params = model.count_params()
print(f"parameters: {n_params:,} (bin table {n_features * args.n_bins * args.embed_dim:,})")

class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.t = time.time()
    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_time_s"] = time.time() - self.t
        logs["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"   epoch {epoch + 1}: {logs['epoch_time_s']:.0f}s, peak RSS {logs['peak_rss_gb']:.1f} GB")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, min_delta=args.min_delta,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_delta=args.min_delta,
                                         min_lr=1e-5, verbose=1),
    tf.keras.callbacks.CSVLogger(os.path.join(out, "history.csv")),
    EpochTimer(),
]

t0 = time.time()
hist = model.fit([Xc_tr, Xb_tr], y_tr, validation_data=([Xc_va, Xb_va], y_va), epochs=args.epochs,
                 batch_size=args.batch_size, class_weight=class_weight, callbacks=callbacks, verbose=2)
train_time = time.time() - t0
best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
print(f"training: {train_time / 60:.1f} min, {len(hist.history['loss'])} epochs, best epoch {best_epoch}")

# ---------------- evaluation (batched) ----------------
p_te = model.predict([Xc_te, Xb_te], batch_size=64, verbose=0)
pred = p_te.argmax(axis=1)
metrics = {
    "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
    "accuracy": float(accuracy_score(y_te, pred)),
    "macro_f1": float(f1_score(y_te, pred, average="macro")),
    "macro_auc_ovr": float(fast_auc(y_te, p_te)),
    "log_loss": float(log_loss(y_te, p_te, labels=list(range(n_classes)))),
    "per_class_f1": dict(zip(classes, f1_score(y_te, pred, average=None).round(4).tolist())),
    "best_epoch": best_epoch, "epochs_trained": len(hist.history["loss"]),
    "train_time_min": round(train_time / 60, 1),
    "mean_epoch_time_s": round(float(np.mean(hist.history["epoch_time_s"])), 1),
    "peak_rss_gb": round(float(max(hist.history["peak_rss_gb"])), 2),
    "n_params": int(n_params),
}
print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_f1"}, indent=1))

# ---------------- save ----------------
model.save_weights(os.path.join(out, "model.weights.h5"))
joblib.dump({"scaler": scaler, "kbd": kbd}, os.path.join(out, "preprocessors.joblib"))
np.savez_compressed(os.path.join(out, "predictions.npz"), p_test=p_te, y_test=y_te, test_idx=test_idx,
                    train_idx=tr_idx, val_idx=va_idx, confusion=confusion_matrix(y_te, pred))
json.dump({"args": vars(args), "metrics": metrics, "classes": classes, "data_dir": args.data_dir},
          open(os.path.join(out, "run.json"), "w"), indent=1)
print(f"saved to {out}/")
