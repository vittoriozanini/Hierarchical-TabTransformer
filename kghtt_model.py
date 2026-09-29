"""
KG-HTT: Hierarchical Knowledge-Graph TabTransformer (Vectorized)
==================================================================
Vectorized pathway processing to handle hundreds of pathways without
the Python-loop graph explosion that killed the original version.

Key changes from v1:
  - Pathways are padded to n_max and processed in batched chunks
  - Attention mask handles padding positions
  - Chunked processing to bound memory: chunk_size pathways at a time
  - Supports mode="sum" and mode="concat" for dual embedding
  - Parameterized multiclass head
  - Pathway-specific and gene-level permutation importance
"""

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
Model = tf.keras.Model


# ======================================================
# TRANSFORMER BLOCK (with mask support)
# ======================================================

class TransformerBlock(layers.Layer):
    """Transformer encoder block with optional padding mask."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate,
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
        padding_mask: (B, T) float32 — 1 for real, 0 for pad.
                      Converted to additive attention mask internally.
        """
        # Convert padding mask to additive attention mask
        if padding_mask is not None:
            # (B, T) -> (B, 1, 1, T) additive mask: 0 for attend, -1e9 for pad
            att_mask = padding_mask[:, None, None, :]  # (B, 1, 1, T)
            att_mask = (1.0 - att_mask) * -1e9  # 0 for real, -1e9 for pad
        else:
            att_mask = None

        attn, scores = self.att(
            x, x, training=training,
            return_attention_scores=True,
            attention_mask=att_mask,
        )
        attn = self.drop1(attn, training=training)
        x1 = self.norm1(x + attn)

        ffn = self.ffn(x1, training=training)
        ffn = self.drop2(ffn, training=training)
        out = self.norm2(x1 + ffn)
        return (out, scores) if return_attention else out


# ======================================================
# DUAL CONTINUOUS EMBEDDING (vectorized)
# ======================================================

class DualContinuousEmbedding(layers.Layer):
    """
    Bin embedding + continuous value projection.
    mode="sum":   e_bin(D) + e_val(D) -> (B, M, D)
    mode="concat": concat(e_bin(D/2), e_val(D/2)) -> (B, M, D)
    """

    def __init__(self, n_features, n_bins, embed_dim,
                 per_feature_proj=True, mode="sum"):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.per_feature_proj = per_feature_proj
        self.mode = mode

        if mode == "concat":
            assert embed_dim % 2 == 0
            self.bin_dim = embed_dim // 2
            self.val_dim = embed_dim // 2
        else:
            self.bin_dim = embed_dim
            self.val_dim = embed_dim

        self.bin_table = self.add_weight(
            shape=(n_features * n_bins, self.bin_dim),
            initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05),
            trainable=True, name="bin_table",
        )
        self.offsets = tf.constant(
            np.arange(n_features, dtype=np.int32) * n_bins
        )

        lim = float(np.sqrt(6.0 / (1 + self.val_dim)))
        w_shape = (n_features, self.val_dim) if per_feature_proj else (1, self.val_dim)
        self.value_w = self.add_weight(
            shape=w_shape,
            initializer=tf.keras.initializers.RandomUniform(-lim, lim),
            trainable=True, name="value_w",
        )
        self.value_b = self.add_weight(
            shape=w_shape, initializer="zeros",
            trainable=True, name="value_b",
        )

    def call(self, x_cont, x_bins):
        e_bin = tf.gather(self.bin_table, tf.cast(x_bins, tf.int32) + self.offsets[None, :])
        e_val = x_cont[:, :, None] * self.value_w[None] + self.value_b[None]
        if self.mode == "concat":
            return tf.concat([e_bin, e_val], axis=-1)
        else:
            return e_bin + e_val


# ======================================================
# HIERARCHICAL TABTRANSFORMER (Vectorized)
# ======================================================

class HierarchicalTabTransformer(Model):
    """
    Vectorized hierarchical TabTransformer.
    Processes pathways in padded, batched chunks to handle hundreds of pathways.
    """

    def __init__(
        self,
        pathway_map,
        n_features,
        n_bins=6,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        local_layers=2,
        global_layers=2,
        dropout=0.1,
        task="binary",
        n_classes=None,
        per_feature_proj=True,
        mode="sum",
        chunk_size=32,
        use_cls_token=False,
    ):
        super().__init__()
        self.pathway_map = pathway_map
        self.task = task
        self.mode = mode
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # --- CLS token (optional) ---
        if use_cls_token:
            self.cls_token = self.add_weight(
                shape=(1, 1, embed_dim),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True, name="cls_token",
            )

        # --- Pre-compute pathway indices and masks ---
        pathway_names = list(pathway_map.keys())
        self.pathway_names = pathway_names
        self.n_pathways = len(pathway_names)
        self.n_max = max(len(idxs) for idxs in pathway_map.values())

        # (K, n_max) — gene indices, padded with 0
        idx_matrix = np.zeros((self.n_pathways, self.n_max), dtype=np.int32)
        # (K, n_max) — 1 for real gene, 0 for padding
        mask_matrix = np.zeros((self.n_pathways, self.n_max), dtype=np.float32)

        for i, name in enumerate(pathway_names):
            idxs = pathway_map[name]
            idx_matrix[i, :len(idxs)] = idxs
            mask_matrix[i, :len(idxs)] = 1.0

        self.pathway_indices = tf.constant(idx_matrix)  # (K, n_max)
        self.pathway_mask = tf.constant(mask_matrix)    # (K, n_max)
        self.pathway_lengths = tf.constant(
            np.array([len(pathway_map[n]) for n in pathway_names], dtype=np.float32)
        )  # (K,)

        # --- Embedding ---
        self.embedding_layer = DualContinuousEmbedding(
            n_features=n_features, n_bins=n_bins, embed_dim=embed_dim,
            per_feature_proj=per_feature_proj, mode=mode,
        )

        # --- Local transformers (shared across pathways) ---
        self.local_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(local_layers)
        ]

        # --- Global transformers ---
        self.global_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(global_layers)
        ]

        # --- Head ---
        if task == "binary":
            out_units, out_act = 1, "sigmoid"
        elif task == "multiclass":
            assert n_classes is not None
            out_units, out_act = n_classes, "softmax"
        else:
            out_units, out_act = 1, "linear"

        self.head = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(out_units, activation=out_act),
        ])

    def _process_pathway_chunk(self, x, pw_indices, pw_mask, training=False):
        """
        Process a chunk of pathways.
        x: (B, M, D) — all gene embeddings
        pw_indices: (chunk_k, n_max) — gene indices for this chunk
        pw_mask: (chunk_k, n_max) — padding mask for this chunk

        Returns: (B, chunk_k, D) — pathway vectors
        """
        B = tf.shape(x)[0]
        chunk_k = tf.shape(pw_indices)[0]
        D = self.embed_dim

        # Gather genes for all pathways in chunk: (B, chunk_k, n_max, D)
        xp = tf.gather(x, pw_indices, axis=1)  # (B, chunk_k, n_max, D)

        # Reshape for batched transformer: (B*chunk_k, n_max, D)
        xp_flat = tf.reshape(xp, [B * chunk_k, self.n_max, D])

        # Build padding mask: (B*chunk_k, n_max)
        pw_mask_flat = tf.tile(pw_mask, [B, 1])  # (B*chunk_k, n_max)

        # Run local transformer blocks
        h = xp_flat
        for blk in self.local_blocks:
            h = blk(h, training=training, padding_mask=pw_mask_flat)

        # Masked mean pool: (B*chunk_k, n_max, D) -> (B*chunk_k, D)
        mask_expanded = pw_mask_flat[:, :, None]  # (B*chunk_k, n_max, 1)
        sum_h = tf.reduce_sum(h * mask_expanded, axis=1)  # (B*chunk_k, D)
        count = tf.reduce_sum(pw_mask_flat, axis=1, keepdims=True)  # (B*chunk_k, 1)
        pathway_vecs = sum_h / (count + 1e-8)  # (B*chunk_k, D)

        # Reshape back: (B, chunk_k, D)
        pathway_vecs = tf.reshape(pathway_vecs, [B, chunk_k, D])
        return pathway_vecs

    def call(self, inputs, training=False):
        x_cont, x_bins = inputs
        x = self.embedding_layer(x_cont, x_bins)  # (B, M, D)

        # Process pathways in chunks
        K = self.n_pathways
        cs = self.chunk_size
        all_vecs = []

        for start in range(0, K, cs):
            end = min(start + cs, K)
            pw_idx = self.pathway_indices[start:end]
            pw_mask = self.pathway_mask[start:end]
            vecs = self._process_pathway_chunk(x, pw_idx, pw_mask, training=training)
            all_vecs.append(vecs)

        z = tf.concat(all_vecs, axis=1)  # (B, K, D)

        # Prepend CLS token if enabled
        if self.use_cls_token:
            B = tf.shape(z)[0]
            cls = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
            z = tf.concat([cls, z], axis=1)  # (B, K+1, D)

        # Global transformer
        h = z
        for blk in self.global_blocks:
            h = blk(h, training=training)

        if self.use_cls_token:
            h = h[:, 0, :]  # (B, D) — read CLS token
        else:
            h = tf.reduce_mean(h, axis=1)  # (B, D)
        return self.head(h)

    # --------------------------------------------------
    # Forward with attention weights
    # --------------------------------------------------
    def forward_with_attention(self, inputs):
        x_cont, x_bins = inputs
        x = self.embedding_layer(x_cont, x_bins)

        K = self.n_pathways
        cs = self.chunk_size
        all_vecs = []
        local_scores = {}

        for start in range(0, K, cs):
            end = min(start + cs, K)
            pw_idx = self.pathway_indices[start:end]
            pw_mask = self.pathway_mask[start:end]
            B = tf.shape(x)[0]
            chunk_k = end - start
            D = self.embed_dim

            xp = tf.gather(x, pw_idx, axis=1)
            xp_flat = tf.reshape(xp, [B * chunk_k, self.n_max, D])
            pw_mask_flat = tf.tile(pw_mask, [B, 1])

            h = xp_flat
            per_layer = []
            for blk in self.local_blocks:
                h, s = blk(h, training=False, return_attention=True, padding_mask=pw_mask_flat)
                per_layer.append(s)
            all_vecs.append(tf.reshape(h, [B, chunk_k, self.n_max, D]))

            for j in range(chunk_k):
                pw_name = self.pathway_names[start + j]
                local_scores[pw_name] = [s[j * B:(j + 1) * B] for s in per_layer]

        z = tf.concat([tf.reduce_mean(v, axis=2) for v in all_vecs], axis=1)

        # Prepend CLS token if enabled
        if self.use_cls_token:
            B = tf.shape(z)[0]
            cls = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
            z = tf.concat([cls, z], axis=1)  # (B, K+1, D)

        h = z
        global_scores = []
        for blk in self.global_blocks:
            h, s = blk(h, training=False, return_attention=True)
            global_scores.append(s)

        if self.use_cls_token:
            # Strip CLS row/column (index 0) from global attention scores
            global_scores = [s[:, :, 1:, 1:] for s in global_scores]
            h = h[:, 0, :]  # (B, D) — read CLS token
        else:
            h = tf.reduce_mean(h, axis=1)

        return self.head(h), local_scores, global_scores

    # --------------------------------------------------
    # Pathway-specific permutation importance
    # --------------------------------------------------
    def forward_with_pathway_permutation(self, inputs, target_pathway, seed=None):
        """Permute only the target pathway's gene embeddings along batch dim."""
        x_cont, x_bins = inputs
        x = self.embedding_layer(x_cont, x_bins)  # (B, M, D)

        B = tf.shape(x)[0]
        if seed is not None:
            perm_idx = tf.random.shuffle(tf.range(B), seed=seed)
        else:
            perm_idx = tf.random.shuffle(tf.range(B))

        x_perm = tf.gather(x, perm_idx, axis=0)  # (B, M, D)

        # Find which chunk and position the target pathway is in
        target_idx = self.pathway_names.index(target_pathway)

        K = self.n_pathways
        cs = self.chunk_size
        all_vecs = []

        for start in range(0, K, cs):
            end = min(start + cs, K)
            pw_idx = self.pathway_indices[start:end]
            pw_mask = self.pathway_mask[start:end]

            # Check if target pathway is in this chunk
            if start <= target_idx < end:
                # Use permuted embeddings for this chunk
                xp = tf.gather(x_perm, pw_idx, axis=1)
            else:
                # Use original embeddings
                xp = tf.gather(x, pw_idx, axis=1)

            chunk_k = end - start
            D = self.embed_dim
            xp_flat = tf.reshape(xp, [B * chunk_k, self.n_max, D])
            pw_mask_flat = tf.tile(pw_mask, [B, 1])

            h = xp_flat
            for blk in self.local_blocks:
                h = blk(h, training=False, padding_mask=pw_mask_flat)

            mask_expanded = pw_mask_flat[:, :, None]
            sum_h = tf.reduce_sum(h * mask_expanded, axis=1)
            count = tf.reduce_sum(pw_mask_flat, axis=1, keepdims=True)
            pathway_vecs = tf.reshape(sum_h / (count + 1e-8), [B, chunk_k, D])
            all_vecs.append(pathway_vecs)

        z = tf.concat(all_vecs, axis=1)
        if self.use_cls_token:
            B = tf.shape(z)[0]
            cls = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
            z = tf.concat([cls, z], axis=1)
        h = z
        for blk in self.global_blocks:
            h = blk(h, training=False)
        if self.use_cls_token:
            return self.head(h[:, 0, :])
        return self.head(tf.reduce_mean(h, axis=1))

    # --------------------------------------------------
    # Gene-level permutation within a pathway
    # --------------------------------------------------
    def forward_with_gene_permutation(self, inputs, target_pathway, gene_idx, seed=None):
        """Permute a single gene's embedding, only for the target pathway."""
        x_cont, x_bins = inputs
        x = self.embedding_layer(x_cont, x_bins)

        B = tf.shape(x)[0]
        if seed is not None:
            perm_idx = tf.random.shuffle(tf.range(B), seed=seed)
        else:
            perm_idx = tf.random.shuffle(tf.range(B))

        # Modified embedding: only column gene_idx is permuted
        x_perm_col = tf.gather(x[:, gene_idx:gene_idx + 1, :], perm_idx, axis=0)
        x_modified = tf.concat([
            x[:, :gene_idx, :], x_perm_col, x[:, gene_idx + 1:, :],
        ], axis=1)

        target_idx = self.pathway_names.index(target_pathway)
        K = self.n_pathways
        cs = self.chunk_size
        all_vecs = []

        for start in range(0, K, cs):
            end = min(start + cs, K)
            pw_idx = self.pathway_indices[start:end]
            pw_mask = self.pathway_mask[start:end]

            if start <= target_idx < end:
                xp = tf.gather(x_modified, pw_idx, axis=1)
            else:
                xp = tf.gather(x, pw_idx, axis=1)

            chunk_k = end - start
            D = self.embed_dim
            xp_flat = tf.reshape(xp, [B * chunk_k, self.n_max, D])
            pw_mask_flat = tf.tile(pw_mask, [B, 1])

            h = xp_flat
            for blk in self.local_blocks:
                h = blk(h, training=False, padding_mask=pw_mask_flat)

            mask_expanded = pw_mask_flat[:, :, None]
            sum_h = tf.reduce_sum(h * mask_expanded, axis=1)
            count = tf.reduce_sum(pw_mask_flat, axis=1, keepdims=True)
            pathway_vecs = tf.reshape(sum_h / (count + 1e-8), [B, chunk_k, D])
            all_vecs.append(pathway_vecs)

        z = tf.concat(all_vecs, axis=1)
        if self.use_cls_token:
            B = tf.shape(z)[0]
            cls = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
            z = tf.concat([cls, z], axis=1)
        h = z
        for blk in self.global_blocks:
            h = blk(h, training=False)
        if self.use_cls_token:
            return self.head(h[:, 0, :])
        return self.head(tf.reduce_mean(h, axis=1))


# ======================================================
# PERMUTATION IMPORTANCE UTILITY
# ======================================================

def pathway_permutation_importance(model, X_cont, X_bins, y,
                                   n_perm=10, seed=0, is_multiclass=False):
    """
    Compute pathway-level permutation importance.
    importance(P) = AUC_baseline - mean(AUC_permuted_P)
    """
    from sklearn.metrics import roc_auc_score

    inputs = [X_cont, X_bins]
    baseline_preds = model(inputs, training=False).numpy()

    if is_multiclass:
        baseline_auc = roc_auc_score(
            y, baseline_preds, multi_class="ovr",
            average="macro", labels=list(range(baseline_preds.shape[1])),
        )
    else:
        baseline_auc = roc_auc_score(y, baseline_preds.ravel())

    results = {}
    for pw_name in model.pathway_names:
        perm_aucs = []
        for p in range(n_perm):
            preds = model.forward_with_pathway_permutation(
                inputs, pw_name, seed=seed + p,
            ).numpy()
            if is_multiclass:
                try:
                    auc = roc_auc_score(
                        y, preds, multi_class="ovr",
                        average="macro", labels=list(range(preds.shape[1])),
                    )
                except ValueError:
                    auc = np.nan
            else:
                auc = roc_auc_score(y, preds.ravel())
            perm_aucs.append(auc)
        results[pw_name] = baseline_auc - np.nanmean(perm_aucs)

    return results, baseline_auc
