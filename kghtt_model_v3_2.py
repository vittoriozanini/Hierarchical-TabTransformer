"""
KG-HTT v3: Hierarchical Knowledge-Graph TabTransformer (vectorized, corrected)
==============================================================================
Changes with respect to Biomni's vectorized version (v3-Biomni):

  [B1] Padding mask passed to MultiHeadAttention as a BOOLEAN mask
       (True = attend). The additive float mask used before was cast to
       bool by Keras and therefore inverted: real genes were masked out.
  [B2] Masked mean pooling is used everywhere (call, forward_with_attention,
       permutation forwards). Before, forward_with_attention averaged over
       padding rows as well.
  [B3] Pathway / gene permutation touches ONLY the target pathway. Before,
       the whole chunk containing the target was permuted.
  [B4] Local attention scores are sliced with the correct stride
       (row index inside a chunk is b * chunk_k + j).
  [P1] Padding positions point to a dedicated all-zero slot (index M),
       not to gene 0.
  [C1] mode="concat" is a full concatenation [e_bin | e_val] in R^{2D}
       followed by a shared linear projection R^{2D} -> R^D, so the
       transformer blocks are identical to mode="sum" and the sum is a
       special case (W = [I; I]).
  [S1] Size-bucketed chunking: pathways sorted by size, each chunk padded
       to its own largest pathway (local cost ~sum n_k^2, not K*n_max^2).
  [E1] Permutation importance with cached pathway vectors: for a joint
       pathway permutation the permuted vector is z[:, t][perm] (no
       re-encoding), then only global blocks + head are re-run.
       Predictions are kept so that permutation SE and bootstrap CI can be
       computed without extra forward passes.
  CLS token removed (kept for a future branch).
"""

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
Model = tf.keras.Model


# ======================================================
# TRANSFORMER BLOCK
# ======================================================

class TransformerBlock(layers.Layer):
    """Pre-activation-free encoder block: MHA -> add&norm -> FFN -> add&norm."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=rate,
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False, return_attention=False, padding_mask=None):
        """
        x: (B, T, D)
        padding_mask: (B, T), 1/True for real tokens, 0/False for padding.
        The key mask is broadcast over the query axis -> boolean (B, T, T).
        """
        att_mask = None
        if padding_mask is not None:
            key_mask = tf.cast(padding_mask, tf.bool)[:, None, :]          # (B, 1, T)
            att_mask = tf.broadcast_to(key_mask, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[1]])

        if return_attention:
            attn, scores = self.att(x, x, training=training,
                                    return_attention_scores=True,
                                    attention_mask=att_mask)
        else:
            attn = self.att(x, x, training=training, attention_mask=att_mask)
            scores = None

        attn = self.drop1(attn, training=training)
        x1 = self.norm1(x + attn)
        ffn = self.ffn(x1, training=training)
        ffn = self.drop2(ffn, training=training)
        out = self.norm2(x1 + ffn)
        return (out, scores) if return_attention else out


# ======================================================
# DUAL CONTINUOUS EMBEDDING
# ======================================================

class DualContinuousEmbedding(layers.Layer):
    """
    Bin embedding + linear projection of the continuous value.
    mode="sum":    e_bin (D) + e_val (D)                          -> (B, M, D)
    mode="concat": [e_bin (D) | e_val (D)] (2D) @ W (2D x D) + b  -> (B, M, D)
    """

    def __init__(self, n_features, n_bins, embed_dim, per_feature_proj=True, mode="sum"):
        super().__init__()
        assert mode in ("sum", "concat")
        self.n_features, self.n_bins, self.embed_dim = n_features, n_bins, embed_dim
        self.per_feature_proj, self.mode = per_feature_proj, mode

        self.bin_table = self.add_weight(
            shape=(n_features * n_bins, embed_dim),
            initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05),
            trainable=True, name="bin_table",
        )
        self.offsets = tf.constant(np.arange(n_features, dtype=np.int32) * n_bins)  # (M,)

        lim = float(np.sqrt(6.0 / (1 + embed_dim)))
        w_shape = (n_features, embed_dim) if per_feature_proj else (1, embed_dim)
        self.value_w = self.add_weight(shape=w_shape,
                                       initializer=tf.keras.initializers.RandomUniform(-lim, lim),
                                       trainable=True, name="value_w")
        self.value_b = self.add_weight(shape=w_shape, initializer="zeros",
                                       trainable=True, name="value_b")

        if mode == "concat":
            # Shared projection R^{2D} -> R^D. With kernel = [I; I] and zero
            # bias this reproduces mode="sum" exactly.
            self.proj = layers.Dense(embed_dim, use_bias=True, name="concat_proj")

    def _combine(self, e_bin, e_val):
        if self.mode == "concat":
            return self.proj(tf.concat([e_bin, e_val], axis=-1))
        return e_bin + e_val

    def call(self, x_cont, x_bins):
        """x_cont (B, M) float32, x_bins (B, M) int -> (B, M, D)"""
        e_bin = tf.gather(self.bin_table, tf.cast(x_bins, tf.int32) + self.offsets[None, :])
        e_val = x_cont[:, :, None] * self.value_w[None] + self.value_b[None]
        return self._combine(e_bin, e_val)

    def call_subset(self, x_cont_sub, x_bins_sub, gene_idx):
        """
        Embed only the genes in gene_idx (1-D int tensor of length n).
        x_cont_sub, x_bins_sub: (B, n), already restricted to gene_idx.
        Returns (B, n, D). Identical to gather(call(...), gene_idx, axis=1).
        """
        gene_idx = tf.cast(gene_idx, tf.int32)
        e_bin = tf.gather(self.bin_table,
                          tf.cast(x_bins_sub, tf.int32) + tf.gather(self.offsets, gene_idx)[None, :])
        if self.per_feature_proj:
            w = tf.gather(self.value_w, gene_idx)
            b = tf.gather(self.value_b, gene_idx)
        else:
            w, b = self.value_w, self.value_b
        e_val = x_cont_sub[:, :, None] * w[None] + b[None]
        return self._combine(e_bin, e_val)


# ======================================================
# HIERARCHICAL TABTRANSFORMER
# ======================================================

class HierarchicalTabTransformer(Model):
    """
    Genes -> local (per-pathway, shared) transformer -> masked mean ->
    pathway vectors z (B, K, D) -> global transformer -> mean -> head.
    Pathways are padded to n_max and processed in chunks of chunk_size.
    """

    def __init__(self, pathway_map, n_features, n_bins=6, embed_dim=64,
                 num_heads=4, ff_dim=128, local_layers=2, global_layers=2,
                 dropout=0.1, task="binary", n_classes=None,
                 per_feature_proj=True, mode="sum", chunk_size=32):
        super().__init__()
        self.pathway_map = pathway_map
        self.task, self.mode = task, mode
        self.chunk_size, self.embed_dim = chunk_size, embed_dim
        self.n_features = n_features

        # ---- pathway index matrices, bucketed by size (padding slot = n_features) ----
        # Pathways are sorted by size and chunked in that order; each chunk is padded
        # only to the size of its largest pathway, so the local attention cost is
        # ~sum(n_k^2) instead of K * n_max^2. Pathway vectors are returned in the
        # ORIGINAL order of pathway_map (self.pathway_names).
        self.pathway_names = list(pathway_map.keys())
        self.n_pathways = len(self.pathway_names)
        self.n_max = max(len(v) for v in pathway_map.values())
        self.pad_index = n_features

        sizes = np.array([len(pathway_map[n]) for n in self.pathway_names])
        self.sorted_order = np.argsort(sizes, kind="stable")            # sorted position -> original index
        self.inverse_order = tf.constant(np.argsort(self.sorted_order).astype(np.int32))  # original -> sorted position
        self.chunks = []                                                # list of (orig_indices, idx (c, n_c), mask (c, n_c))
        for start in range(0, self.n_pathways, chunk_size):
            orig = self.sorted_order[start:start + chunk_size]
            n_c = int(sizes[orig].max())
            idx = np.full((len(orig), n_c), self.pad_index, dtype=np.int32)
            msk = np.zeros((len(orig), n_c), dtype=np.float32)
            for j, oi in enumerate(orig):
                g = np.asarray(pathway_map[self.pathway_names[oi]], dtype=np.int32)
                idx[j, :len(g)], msk[j, :len(g)] = g, 1.0
            self.chunks.append((orig, tf.constant(idx), tf.constant(msk)))

        # ---- layers ----
        self.embedding_layer = DualContinuousEmbedding(
            n_features, n_bins, embed_dim, per_feature_proj, mode)
        self.local_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                             for _ in range(local_layers)]
        self.global_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                              for _ in range(global_layers)]

        if task == "binary":
            out_units, out_act = 1, "sigmoid"
        elif task == "multiclass":
            assert n_classes is not None
            out_units, out_act = n_classes, "softmax"
        else:
            out_units, out_act = 1, "linear"
        self.head = tf.keras.Sequential([
            layers.Dense(128, activation="relu"), layers.Dropout(dropout),
            layers.Dense(64, activation="relu"), layers.Dropout(dropout),
            layers.Dense(out_units, activation=out_act),
        ])

    # --------------------------------------------------
    # building blocks
    # --------------------------------------------------
    def _embed_with_pad(self, x_cont, x_bins):
        """(B, M) -> (B, M+1, D); last row is the all-zero padding slot."""
        x = self.embedding_layer(x_cont, x_bins)
        zeros = tf.zeros([tf.shape(x)[0], 1, self.embed_dim], dtype=x.dtype)
        return tf.concat([x, zeros], axis=1)

    @staticmethod
    def _masked_mean(h, mask):
        """h (N, T, D), mask (N, T) -> (N, D)"""
        m = mask[:, :, None]
        return tf.reduce_sum(h * m, axis=1) / tf.reduce_sum(m, axis=1)

    def _run_local(self, seq, mask, training=False, return_attention=False):
        """seq (N, T, D), mask (N, T) -> pooled (N, D) [, list of scores]"""
        h, per_layer = seq, []
        for blk in self.local_blocks:
            if return_attention:
                h, s = blk(h, training=training, return_attention=True, padding_mask=mask)
                per_layer.append(s)
            else:
                h = blk(h, training=training, padding_mask=mask)
        pooled = self._masked_mean(h, mask)
        return (pooled, per_layer) if return_attention else pooled

    def _pathway_vectors(self, x, training=False, return_attention=False):
        """
        x (B, M+1, D) -> z (B, K, D). Chunked over pathways.
        If return_attention, also returns {pathway: [scores per layer (B, H, n_k, n_k)]}.
        """
        B = tf.shape(x)[0]
        D = self.embed_dim
        all_vecs, local_scores = [], {}
        for orig, pw_idx, pw_mask in self.chunks:                # size-bucketed chunks
            c, n_c = pw_idx.shape
            xp = tf.gather(x, pw_idx, axis=1)                    # (B, c, n_c, D)
            xp_flat = tf.reshape(xp, [B * c, n_c, D])            # row = b*c + j
            mask_flat = tf.tile(pw_mask, [B, 1])                 # row = b*c + j

            if return_attention:
                pooled, per_layer = self._run_local(xp_flat, mask_flat, training, True)
                for j, oi in enumerate(orig):
                    name = self.pathway_names[oi]
                    n_k = len(self.pathway_map[name])
                    local_scores[name] = [s[j::c][:, :, :n_k, :n_k] for s in per_layer]
            else:
                pooled = self._run_local(xp_flat, mask_flat, training)
            all_vecs.append(tf.reshape(pooled, [B, c, D]))

        z_sorted = tf.concat(all_vecs, axis=1)                   # (B, K, D) in sorted order
        z = tf.gather(z_sorted, self.inverse_order, axis=1)      # back to original order
        return (z, local_scores) if return_attention else z

    def _global_and_head(self, z, training=False, return_attention=False):
        h, global_scores = z, []
        for blk in self.global_blocks:
            if return_attention:
                h, s = blk(h, training=training, return_attention=True)
                global_scores.append(s)
            else:
                h = blk(h, training=training)
        out = self.head(tf.reduce_mean(h, axis=1), training=training)
        return (out, global_scores) if return_attention else out

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def call(self, inputs, training=False):
        x_cont, x_bins = inputs
        x = self._embed_with_pad(x_cont, x_bins)
        z = self._pathway_vectors(x, training=training)
        return self._global_and_head(z, training=training)

    def forward_with_attention(self, inputs):
        """Returns (predictions, local_scores dict, list of global scores (B, H, K, K))."""
        x_cont, x_bins = inputs
        x = self._embed_with_pad(x_cont, x_bins)
        z, local_scores = self._pathway_vectors(x, training=False, return_attention=True)
        out, global_scores = self._global_and_head(z, training=False, return_attention=True)
        return out, local_scores, global_scores

    # --------------------------------------------------
    # cached-pathway inference (used by permutation importance)
    # --------------------------------------------------
    def compute_pathway_vectors(self, x_cont, x_bins):
        """(B, M) inputs -> z (B, K, D), no dropout."""
        x = self._embed_with_pad(x_cont, x_bins)
        return self._pathway_vectors(x, training=False)

    def encode_pathway_subset(self, x_cont, x_bins, target_idx, perm_idx=None, gene_pos=None):
        """
        Re-encode ONLY pathway `target_idx` from raw inputs.
        perm_idx: (B,) permutation of the batch axis applied to the genes of
                  the target pathway (all genes jointly) or, if gene_pos is
                  given, only to the gene in position gene_pos of the pathway.
        Returns the pathway vector (B, D).
        """
        name = self.pathway_names[target_idx]
        gene_idx = tf.constant(np.asarray(self.pathway_map[name], dtype=np.int32))
        xc = tf.gather(x_cont, gene_idx, axis=1)
        xb = tf.gather(x_bins, gene_idx, axis=1)
        e = self.embedding_layer.call_subset(xc, xb, gene_idx)     # (B, n_k, D)
        if perm_idx is not None:
            perm_idx = tf.cast(perm_idx, tf.int32)
            if gene_pos is None:
                e = tf.gather(e, perm_idx, axis=0)
            else:
                col = tf.gather(e[:, gene_pos:gene_pos + 1, :], perm_idx, axis=0)
                e = tf.concat([e[:, :gene_pos, :], col, e[:, gene_pos + 1:, :]], axis=1)
        mask = tf.ones([tf.shape(e)[0], tf.shape(e)[1]], dtype=tf.float32)
        return self._run_local(e, mask, training=False)

    def predict_from_pathway_vectors(self, z):
        """z (B, K, D) -> predictions, no dropout."""
        return self._global_and_head(z, training=False)

    def forward_with_pathway_permutation(self, inputs, target_pathway, perm_idx, z_cache=None):
        """Full-batch convenience wrapper: permute only `target_pathway`."""
        x_cont, x_bins = inputs
        t = self.pathway_names.index(target_pathway)
        z = self.compute_pathway_vectors(x_cont, x_bins) if z_cache is None else z_cache
        return self.predict_from_pathway_vectors(self._replace_column(
            z, t, self.encode_pathway_subset(x_cont, x_bins, t, perm_idx)))

    def forward_with_gene_permutation(self, inputs, target_pathway, gene_idx, perm_idx, z_cache=None):
        """Permute one gene (global index gene_idx) only inside `target_pathway`."""
        x_cont, x_bins = inputs
        t = self.pathway_names.index(target_pathway)
        pos = list(self.pathway_map[target_pathway]).index(gene_idx)
        z = self.compute_pathway_vectors(x_cont, x_bins) if z_cache is None else z_cache
        return self.predict_from_pathway_vectors(self._replace_column(
            z, t, self.encode_pathway_subset(x_cont, x_bins, t, perm_idx, gene_pos=pos)))

    @staticmethod
    def _replace_column(z, t, vec):
        """z (B, K, D), vec (B, D) -> z with column t replaced."""
        return tf.concat([z[:, :t, :], vec[:, None, :], z[:, t + 1:, :]], axis=1)


# ======================================================
# FAST AUC (rank based, macro one-vs-rest)
# ======================================================

def fast_auc(y, scores):
    """
    y: (n,) int labels; scores: (n, C) probabilities, or (n,) / (n, 1) for binary.
    Returns macro OVR AUC (classes absent from y are skipped). Matches
    sklearn.metrics.roc_auc_score(..., multi_class="ovr", average="macro").
    """
    from scipy.stats import rankdata
    scores = np.asarray(scores)
    if scores.ndim == 1 or scores.shape[1] == 1:
        scores = np.column_stack([1 - scores.ravel(), scores.ravel()])
    n, C = scores.shape
    ranks = rankdata(scores, axis=0)
    aucs = []
    for c in range(C):
        pos = (y == c)
        n_pos, n_neg = pos.sum(), n - pos.sum()
        if n_pos == 0 or n_neg == 0:
            continue
        aucs.append((ranks[pos, c].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs))


# ======================================================
# PERMUTATION IMPORTANCE WITH CACHED PATHWAY VECTORS
# ======================================================

def _batched(fn, n, batch_size):
    return np.concatenate([fn(s, min(s + batch_size, n)) for s in range(0, n, batch_size)], axis=0)


def pathway_permutation_importance(model, X_cont, X_bins, y, n_perm=20, seed=0,
                                   batch_size=64, n_boot=200, boot_seed=0,
                                   pathways=None, return_predictions=False, verbose=True):
    """
    importance(P) = AUC_baseline - mean_k AUC(y, pred_k(P)), where pred_k(P) is
    obtained by permuting the genes of P (jointly) along the sample axis, only
    inside P, and re-running the global blocks + head on cached pathway vectors.

    Returns dict pathway -> {importance, se_perm, ci_low, ci_high, auc_perm_mean}
    plus baseline AUC. se_perm = sd of the n_perm AUCs / sqrt(n_perm).
    ci = percentile bootstrap (n_boot resamples of the test samples, paired
    between baseline and permuted predictions). n_boot=0 disables it.
    """
    import time
    X_cont = np.asarray(X_cont, dtype=np.float32)
    X_bins = np.asarray(X_bins, dtype=np.int32)
    y = np.asarray(y).astype(int)
    n = len(y)
    rng = np.random.RandomState(seed)
    brng = np.random.RandomState(boot_seed)
    boot_idx = [brng.randint(0, n, n) for _ in range(n_boot)]

    t0 = time.time()
    z = _batched(lambda s, e: model.compute_pathway_vectors(X_cont[s:e], X_bins[s:e]).numpy(),
                 n, batch_size)                                                    # (n, K, D)
    p0 = _batched(lambda s, e: model.predict_from_pathway_vectors(tf.constant(z[s:e])).numpy(),
                  n, batch_size)                                                    # (n, C)
    auc0 = fast_auc(y, p0)
    if verbose:
        print(f"cached pathway vectors {z.shape} in {time.time() - t0:.1f}s, baseline AUC {auc0:.4f}")

    perms = [rng.permutation(n) for _ in range(n_perm)]   # same permutations for every pathway
    names = pathways if pathways is not None else model.pathway_names
    results, preds_store = {}, {}
    for i, name in enumerate(names):
        t = model.pathway_names.index(name)
        t1 = time.time()
        aucs, preds = [], []
        # Joint permutation of the genes of P along the sample axis commutes with
        # the (per-sample) local encoding: encode(x[perm]) == encode(x)[perm].
        # So no re-encoding is needed: the permuted pathway vector is z[:, t][perm].
        # (Re-encoding via encode_pathway_subset is required only for single-gene
        # permutations, where samples are mixed inside the pathway.)
        for perm in perms:
            z_mod = z.copy()
            z_mod[:, t, :] = z[perm, t, :]
            pk = _batched(lambda s, e: model.predict_from_pathway_vectors(tf.constant(z_mod[s:e])).numpy(),
                          n, batch_size)
            aucs.append(fast_auc(y, pk))
            preds.append(pk)
        aucs = np.array(aucs)
        res = {"importance": float(auc0 - aucs.mean()),
               "se_perm": float(aucs.std(ddof=1) / np.sqrt(n_perm)) if n_perm > 1 else np.nan,
               "auc_perm_mean": float(aucs.mean())}
        if n_boot > 0:
            diffs = []
            for bi in boot_idx:
                a0 = fast_auc(y[bi], p0[bi])
                ak = np.mean([fast_auc(y[bi], pk[bi]) for pk in preds])
                diffs.append(a0 - ak)
            res["ci_low"], res["ci_high"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
        results[name] = res
        if return_predictions:
            preds_store[name] = np.stack(preds)
        if verbose:
            print(f"[{i + 1}/{len(names)}] {name[:50]:50s} I={res['importance']:+.4f} "
                  f"SE={res['se_perm']:.4f} ({time.time() - t1:.1f}s)")
    if return_predictions:
        return results, auc0, {"baseline": p0, "permuted": preds_store}
    return results, auc0
