"""
KG-HTT v3: Hierarchical TabTransformer
=========================================================
B: samples in the batch (batch_size)
M: number of genes (n_features)
K: number of pathways
D: embedding dimension (embed_dim)
n_max: maximum number of genes in a pathway (the rest are padded)
c: number of pathways in a chunk (chunk_size)
T: sequence length (n_max for local, K for global)
H: number of attention heads (num_heads)

Input:
    x_cont: (B, M) float32, continuous values
    x_bins: (B, M) int, binned values in [0, n_bins-1]
"""

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
Model = tf.keras.Model

# ======================================================
# TRANSFORMER BLOCK
# ======================================================

class TransformerBlock(layers.Layer):
    """
    MHA -> add&norm -> FFN -> add&norm
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        
        # --- Multi-Head Attention layer. key_dim is the size of each attention head. Dropout is applied to the attention scores ---
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=rate,
        )
        
        # --- Feed-Forward Network (FFN) consists of two Dense layers with a GELU activation in between. Dropout is applied after the first Dense layer ---
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        
        # --- LayerNorm: normalizes the inputs across the features ---
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False, return_attention=False, padding_mask=None):
        """
        Input x: (B, T, D)
        padding_mask: (B, T), 1/True for real tokens, 0/False for padding
        Returns (B, T, D) or ((B, T, D), list of attention scores per layer) if return_attention is True
        """
        att_mask = None
        if padding_mask is not None:
            key_mask = tf.cast(padding_mask, tf.bool)[:, None, :]          # (B, T) -> (B, 1, T)
            att_mask = tf.broadcast_to(key_mask, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[1]]) # (B, T, T)

        # --- Self-attention: queries, keys, and values are all the same input x --- 
        if return_attention:
            attn, scores = self.att(x, x, training=training,
                                    return_attention_scores=True,
                                    attention_mask=att_mask)
        else:
            attn = self.att(x, x, training=training, attention_mask=att_mask)
            scores = None

        
        attn = self.drop1(attn, training=training)  # (B, T, D)
        x1 = self.norm1(x + attn)                   # (B, T, D)     x1 = LayerNorm(x + Dropout(MHA(x, x)))
        ffn = self.ffn(x1, training=training)       # (B, T, D)
        ffn = self.drop2(ffn, training=training)    # (B, T, D)
        out = self.norm2(x1 + ffn)                  # (B, T, D)     out = LayerNorm(x1 + Dropout(FFN(x1)))
        return (out, scores) if return_attention else out


# ======================================================
# DUAL CONTINUOUS EMBEDDING
# ======================================================

class DualContinuousEmbedding(layers.Layer):
    """
    Bin embedding + linear projection of the continuous value.
    mode="sum":    e_bin (D) + e_val (D)                          -> (B, M, D)
    mode="concat": [e_bin (D) | e_val (D)] (2D) @ W (2D x D) + b  -> (B, M, D)
    Each feature has its own linear projection (per_feature_proj=True) or they share one (per_feature_proj=False).
    """

    def __init__(self, n_features, n_bins, embed_dim, per_feature_proj=True, mode="sum"):
        super().__init__()
        assert mode in ("sum", "concat")
        self.n_features, self.n_bins, self.embed_dim = n_features, n_bins, embed_dim
        self.per_feature_proj, self.mode = per_feature_proj, mode

        # --- Bin embedding table: (n_features * n_bins, embed_dim). Each feature has its own bin embeddings ---
        self.bin_table = self.add_weight(
            shape=(n_features * n_bins, embed_dim),
            initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05),
            trainable=True, name="bin_table",
        )
        # offsets: (M,) = [0, n_bins, 2*n_bins, ..., (M-1)*n_bins], used to index into the bin_table for each feature
        self.offsets = tf.constant(np.arange(n_features, dtype=np.int32) * n_bins)  # (M,)
 
        # --- Linear projection of continuous values: e_val = x_cont * w + b ---
        # w (M, D) and b (M, D) if per_feature_proj = True else w (1, D) and b (1, D)
        lim = float(np.sqrt(6.0 / (1 + embed_dim)))
        w_shape = (n_features, embed_dim) if per_feature_proj else (1, embed_dim)
        self.value_w = self.add_weight(shape=w_shape,
                                       initializer=tf.keras.initializers.RandomUniform(-lim, lim),
                                       trainable=True, name="value_w")
        self.value_b = self.add_weight(shape=w_shape, initializer="zeros",
                                       trainable=True, name="value_b")

        if mode == "concat":
            # Shared projection R^{2D} -> R^D. With kernel = [I; I] and zero
            # bias this reproduces mode="sum" exactly
            self.proj = layers.Dense(embed_dim, use_bias=True, name="concat_proj")

    def _combine(self, e_bin, e_val):
        if self.mode == "concat":
            return self.proj(tf.concat([e_bin, e_val], axis=-1)) # concatenate along the last dimension and project to embed_dim (B, M, 2D) -> (B, M, D)
        return e_bin + e_val # sum the bin embedding and the value embedding (B, M, D) + (B, M, D) -> (B, M, D)


    def call(self, x_cont, x_bins):
        """x_cont (B, M) float32, x_bins (B, M) int -> (B, M, D)"""
        # x_bins (B, M) + offsets[None, :] (1, M) -> (B, M) int32, then gather from bin_table (M * n_bins, D) -> (B, M, D)
        e_bin = tf.gather(self.bin_table, tf.cast(x_bins, tf.int32) + self.offsets[None, :])
        e_val = x_cont[:, :, None] * self.value_w[None] + self.value_b[None]
        return self._combine(e_bin, e_val)

    def call_subset(self, x_cont_sub, x_bins_sub, gene_idx):
        """
        Same as call(), but only for a subset of genes (gene_idx). This is used for permutation importance.
        Embed only the genes in gene_idx (1-D int tensor of length n).
        x_cont_sub, x_bins_sub: (B, n), already restricted to gene_idx.
        Returns (B, n, D).
        """
        gene_idx = tf.cast(gene_idx, tf.int32)
        e_bin = tf.gather(self.bin_table,
                          tf.cast(x_bins_sub, tf.int32) + tf.gather(self.offsets, gene_idx)[None, :])
        if self.per_feature_proj:
            w = tf.gather(self.value_w, gene_idx)              # (n, D)
            b = tf.gather(self.value_b, gene_idx)              # (n, D)
        else:
            w, b = self.value_w, self.value_b                  # (1, D)
        e_val = x_cont_sub[:, :, None] * w[None] + b[None]     # (B, n, D)
        return self._combine(e_bin, e_val)


# ======================================================
# HIERARCHICAL TABTRANSFORMER
# ======================================================

class HierarchicalTabTransformer(Model):
    """
    Genes -> local (per-pathway, shared weights) transformer -> masked mean ->
    pathway vectors z (B, K, D) -> global transformer -> mean -> head.
    Pathways are sorted by size and processed in chunks of chunk_size to limit memory usage.
    Each chunk is padded to the maximum size of the pathways in that chunk.
    """

    def __init__(self, pathway_map, n_features, n_bins=6, embed_dim=64,
                 num_heads=4, ff_dim=128, local_layers=2, global_layers=2,
                 dropout=0.1, task="binary", n_classes=None,
                 per_feature_proj=True, mode="sum", chunk_size=32):
        super().__init__()
        self.pathway_map = pathway_map # dict {pathway_name: [list of gene indices]}
        self.task, self.mode = task, mode
        self.chunk_size, self.embed_dim = chunk_size, embed_dim
        self.n_features = n_features

        # --- pathway index matrix, bucketed by size and mask (padding slot = n_features) ---
        # pathways are sorted by size, then split into chunks of chunk_size pathways. Each chunk is padded to the maximum size of the pathways in that chunk.
        self.pathway_names = list(pathway_map.keys())
        self.n_pathways = len(self.pathway_names) # K
        self.n_max = max(len(v) for v in pathway_map.values())
        self.pad_index = n_features
        
        
        sizes = np.array([len(pathway_map[n]) for n in self.pathway_names]) # (K,) number of genes in each pathway
        self.sorted_order = np.argsort(sizes, kind="stable") # (K,) indices of pathways sorted by size
        self.inverse_order = tf.constant(np.argsort(self.sorted_order).astype(np.int32)) # (K,) indices to invert the sorting
        self.chunks = [] # list of (oirg_idx, idx (c, n_c), mask (c, n_c))
        
        for start in range(0, self.n_pathways, self.chunk_size):
            orig = self.sorted_order[start:start + self.chunk_size] # original indices of the pathways in this chunk
            n_c = int(sizes[orig].max()) # maximum number of genes in this chunk
            idx = np.full((len(orig), n_c), self.pad_index, dtype=np.int32) # (c, n_c) filled with pad_index
            msk = np.zeros((len(orig), n_c), dtype=np.float32) # (c, n_c) filled with zeros
            for j, oi in enumerate(orig):
                g = np.asarray(pathway_map[self.pathway_names[oi]], dtype=np.int32) # genes of pathway oi
                idx[j, :len(g)], msk[j, :len(g)] = g, 1.0 # first len(g) entries are the gene indices, rest are pad_index; mask is 1 for real genes, 0 for padding
            self.chunks.append((orig, tf.constant(idx), tf.constant(msk))) # store the original indices, the index matrix, and the mask for this chunk


        # --- layers ---
        self.embedding_layer = DualContinuousEmbedding(
            n_features, n_bins, embed_dim, per_feature_proj, mode)
        self.local_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) # local blocks are shared across pathways
                             for _ in range(local_layers)] 
        self.global_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) # global blocks are applied to the pathway vectors z (B, K, D)
                              for _ in range(global_layers)]
        
        # --- output head, based on task ---
        if task == "binary":
            out_units, out_act = 1, "sigmoid" # (B, 1)
        elif task == "multiclass":
            assert n_classes is not None
            out_units, out_act = n_classes, "softmax" # (B, n_classes)
        else:
            out_units, out_act = 1, "linear" # regression (B, 1)
            
        # MLP head: Dense -> Dropout -> Dense -> Dropout -> Dense
        # D -> 128 -> 64 -> out_units    
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
        x = self.embedding_layer(x_cont, x_bins) # (B, M, D)
        zeros = tf.zeros([tf.shape(x)[0], 1, self.embed_dim], dtype=x.dtype) # (B, 1, D)
        return tf.concat([x, zeros], axis=1) # (B, M+1, D)
        # x[:, self.pad_index, :] = 0.0 # last row is the all-zero padding slot
    
    
    @staticmethod
    def _masked_mean(h, mask):
        """
        h (N, T, D), mask (N, T) -> (N, D)
        Computes the mean of h over the T dimension, ignoring the padded positions (where mask=0) / selects only the valid positions (where mask=1).
        """
        m = mask[:, :, None] # (N, T) -> (N, T, 1)
        return tf.reduce_sum(h * m, axis=1) / tf.reduce_sum(m, axis=1)
        # numerator: sum of h over T, weighted by mask (only valid positions contribute)
        # denominator: sum of mask over T (number of valid positions)
        # Note: if all positions are padded (mask=0), this will result in NaN. In practice, this should not happen if the input is valid


    def _run_local(self, seq, mask, training=False, return_attention=False):
        """
        seq (N, T, D), mask (N, T) -> pooled (N, D) [, list of scores per layer (N, H, T, T)] if return_attention is True.
        N is generic, in the forward pass it is N = B * c (batch size times number of pathways in the chunk).
        In permutation importance, N is just the batch size B (we only process one pathway at a time).
        """
        h, per_layer = seq, []
        for blk in self.local_blocks: 
            if return_attention:
                h, s = blk(h, training=training, return_attention=True, padding_mask=mask)
                per_layer.append(s) # (N, H, T, T) attention scores for this layer
            else:
                h = blk(h, training=training, padding_mask=mask)
        pooled = self._masked_mean(h, mask) # (N, T, D) -> (N, D) a vector representation for each pathway, pooled over the genes in the pathway
        return (pooled, per_layer) if return_attention else pooled

    def _pathway_vectors(self, x, training=False, return_attention=False):
        """
        x (B, M+1, D) -> z (B, K, D). Pathways are sorted by size and processed in chunks of chunk_size to limit memory usage.
        The resulting z is reordered back to the original order of the pathways.
        If return_attention, also returns {pathway: [scores per layer (B, H, n_k, n_k)]}.
        """
        B = tf.shape(x)[0]
        D = self.embed_dim
        all_vecs, local_scores = [], {}
        
        for orig, pw_idx, pw_mask in self.chunks: # size-bucketed chunks of pathways, each with its own index matrix and mask
            c, n_c = pw_idx.shape # number of pathways in this chunk, maximum number of genes in this chunk
            xp = tf.gather(x, pw_idx, axis=1)                    # (B, c, n_c, D) gather the embeddings of the genes in the pathways for this chunk
            xp_flat = tf.reshape(xp, [B * c, n_c, D])            # (B*c, n_c, D), row = b*c + j, merge axes B and c, each pathway is now a separate sequence in the batch
            mask_flat = tf.tile(pw_mask, [B, 1])                 # (B*c, n_c), row = b*c + j, the mask (c, n_c) is repeated B times for each sample in the batch
        

            if return_attention:
                pooled, per_layer = self._run_local(xp_flat, mask_flat, training, True)

                for j, oi in enumerate(orig): # for each pathway in the chunk
                    name = self.pathway_names[oi]
                    n_k = len(self.pathway_map[name])
                    # pathway j rows are j, j+c, j+2c, ..., j+(B-1)c in the flattened batch of size B*c
                    # slicing s[j::c] selects the rows corresponding to pathway j across all B samples -> (B, H, n_max, n_max)
                    # then we slice [:, :, :n_k, :n_k] to get the attention scores for the valid genes in the pathway (ignoring padding) -> (B, H, n_k, n_k)
                    local_scores[name] = [s[j::c][:, :, :n_k, :n_k] for s in per_layer]
            else:
                pooled = self._run_local(xp_flat, mask_flat, training) # (B*c, D)
                
            all_vecs.append(tf.reshape(pooled, [B, c, D])) # invert the flattening to get back to (B, c, D) for this chunk of pathways
            
            
        z_sorted = tf.concat(all_vecs, axis=1) # (B, c_i, D) -> (B, K, D) concatenate all chunks along the pathway axis, in sorted order of pathway size
        z = tf.gather(z_sorted, self.inverse_order, axis=1) # (B, K, D) reorder the pathways back to the original order using the inverse of the sorted order
        return (z, local_scores) if return_attention else z

    def _global_and_head(self, z, training=False, return_attention=False):
        """ z (B, K, D) -> predictions (B, out_units) [, list of global attention scores (B, H, K, K)] if return_attention is True."""
        h, global_scores = z, []
        for blk in self.global_blocks:
            if return_attention:
                h, s = blk(h, training=training, return_attention=True)
                global_scores.append(s) # s: (B, H, K, K) attention scores for this layer, pathways attend to each other
            else:
                h = blk(h, training=training) # (B, K, D)
        out = self.head(tf.reduce_mean(h, axis=1), training=training) # average over the K pathways to get a single vector per sample (B, K, D) -> (B, D) -> head -> (B, out_units)
        return (out, global_scores) if return_attention else out

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def call(self, inputs, training=False):
        x_cont, x_bins = inputs # (B, M) and (B, M) continuous and binned inputs
        x = self._embed_with_pad(x_cont, x_bins) # (B, M+1, D) embedding with padding slot
        z = self._pathway_vectors(x, training=training) # (B, K, D) pathway vectors
        return self._global_and_head(z, training=training)

    def forward_with_attention(self, inputs):
        """
        Same as call(), but returns local and global attention scores for interpretability.
        local_scores: dictionary {pathway_name: list of attention scores per local layer [(B, H, n_k, n_k)]}
        global_scores: list of attention scores per global layer [(B, H, K, K)]
        """
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
        Returns the permuted pathway vector (B, D).
        """
        name = self.pathway_names[target_idx]
        gene_idx = tf.constant(np.asarray(self.pathway_map[name], dtype=np.int32)) # (n_k,) global indices of the genes in the target pathway
        xc = tf.gather(x_cont, gene_idx, axis=1)
        xb = tf.gather(x_bins, gene_idx, axis=1)
        e = self.embedding_layer.call_subset(xc, xb, gene_idx)     # (B, n_k, D)
        
        if perm_idx is not None:
            perm_idx = tf.cast(perm_idx, tf.int32) 
            if gene_pos is None:
                # permute all genes in the pathway jointly
                e = tf.gather(e, perm_idx, axis=0) # (B, n_k, D) permute the batch axis of the pathway embeddings
                
            else:
                # permute only the gene at position gene_pos in the pathway
                col = tf.gather(e[:, gene_pos:gene_pos + 1, :], perm_idx, axis=0) # (B, 1, D) permute the batch axis of the single gene embedding
                e = tf.concat([e[:, :gene_pos, :], col, e[:, gene_pos + 1:, :]], axis=1) # (B, n_k, D) replace the original gene embedding with the permuted one
                
        # No padding is needed here because the local transformer will handle the variable-length sequences using the mask. 
        # The mask will be all ones for the valid genes in the pathway, and zeros for any padding (if applicable). 
        mask = tf.ones([tf.shape(e)[0], tf.shape(e)[1]], dtype=tf.float32) # (B, n_k) mask indicating that all positions in e are valid (no padding)
        return self._run_local(e, mask, training=False) # (B, D) pathway vector for the target pathway, after local transformer and pooling

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
        """
        z (B, K, D), vec (B, D) -> z with column t replaced by vec.
        TF tensors are immutable, so we need to create a new tensor with the updated column. 
        We can do this by concatenating slices of z before and after column t with vec in between.
        """
        return tf.concat([z[:, :t, :], vec[:, None, :], z[:, t + 1:, :]], axis=1)


# ======================================================
# FAST AUC (rank based, macro one-vs-rest)
# ======================================================

def fast_auc(y, scores):
    """
    y: (n,) int labels; scores: (n, C) probabilities, or (n,) / (n, 1) for binary.
    Returns macro OVR AUC (classes absent from y are skipped).
    This is equivalent to sklearn.metrics.roc_auc_score(..., multi_class="ovr", average="macro").
    """
    from scipy.stats import rankdata
    scores = np.asarray(scores)
    if scores.ndim == 1 or scores.shape[1] == 1: # binary case, convert to (n, 2) with scores for class 0 and class 1
        scores = np.column_stack([1 - scores.ravel(), scores.ravel()])
    n, C = scores.shape
    ranks = rankdata(scores, axis=0) 
    aucs = []
    for c in range(C):
        pos = (y == c) # boolean array of shape (n,) indicating which samples belong to class c
        n_pos, n_neg = pos.sum(), n - pos.sum()
        if n_pos == 0 or n_neg == 0: # if there are no positive or no negative samples for this class, we cannot compute AUC, so we skip it
            continue
        
        # Mann-Whitney U = sum of ranks of positive samples - n_pos * (n_pos + 1) / 2
        # AUC = U / (n_pos * n_neg)
        aucs.append((ranks[pos, c].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs))


# ======================================================
# PERMUTATION IMPORTANCE WITH CACHED PATHWAY VECTORS
# ======================================================

def _batched(fn, n, batch_size):
    """
    Run fn(s, e) for s=0, batch_size, 2*batch_size, ..., n and concatenate the results along axis=0.
    This is useful for processing large datasets in batches to avoid memory issues.
    """
    
    return np.concatenate([fn(s, min(s + batch_size, n)) for s in range(0, n, batch_size)], axis=0)


def pathway_permutation_importance(model, X_cont, X_bins, y, n_perm=20, seed=0,
                                   batch_size=64, n_boot=200, boot_seed=0,
                                   pathways=None, return_predictions=False, verbose=True):
    """
    importance(P) = AUC_baseline - mean_k AUC(y, pred_k(P)), where pred_k(P) is
    obtained by permuting the genes of P (jointly) along the sample axis, only
    inside P, and re-running the global blocks + head on cached pathway vectors.

    Returns dict pathway -> {importance, se_perm, ci_low, ci_high, auc_perm_mean} plus baseline AUC. 
    se_perm = sd of the n_perm AUCs / sqrt(n_perm).
    ci = percentile bootstrap (n_boot resamples of the test samples, paired
    between baseline and permuted predictions). n_boot=0 disables it.
    """
    import time
    X_cont = np.asarray(X_cont, dtype=np.float32)
    X_bins = np.asarray(X_bins, dtype=np.int32)
    y = np.asarray(y).astype(int)
    n = len(y)
    rng = np.random.RandomState(seed) # random number generator for permutations
    brng = np.random.RandomState(boot_seed) # random number generator for bootstrap resampling
    boot_idx = [brng.randint(0, n, n) for _ in range(n_boot)]
    # n_boot resamples of the indices of the test samples, each of size n, sampled with replacement

    # --- cache the pathway vectors and baseline predictions ---
    t0 = time.time()
    z = _batched(lambda s, e: model.compute_pathway_vectors(X_cont[s:e], X_bins[s:e]).numpy(),
                 n, batch_size)                                                    # (n, K, D)
    p0 = _batched(lambda s, e: model.predict_from_pathway_vectors(tf.constant(z[s:e])).numpy(),
                  n, batch_size)                                                    # (n, C)
    auc0 = fast_auc(y, p0)
    if verbose:
        print(f"cached pathway vectors {z.shape} in {time.time() - t0:.1f}s, baseline AUC {auc0:.4f}")

    # --- generate permutations for each pathway ---
    perms = [rng.permutation(n) for _ in range(n_perm)]   # same permutations for every pathway
    names = pathways if pathways is not None else model.pathway_names
    results, preds_store = {}, {}
    for i, name in enumerate(names):
        t = model.pathway_names.index(name)
        t1 = time.time()
        aucs, preds = [], []
        
        # Joint permutation of the genes of P along the sample axis commutes with
        # the (per-sample) local encoding: encode(x[perm]) == encode(x)[perm].
        # So no re-encoding is needed, the permuted pathway vector is z[:,t][perm].
        # Re-encoding is only needed if we permute a single gene inside the pathway, which is not done here.
        for perm in perms:
            z_mod = z.copy()                                           # (n, K, D)
            z_mod[:, t, :] = z[perm, t, :]
            pk = _batched(lambda s, e: model.predict_from_pathway_vectors(tf.constant(z_mod[s:e])).numpy(),
                          n, batch_size)                            # (n, C) predictions with permuted pathway
            aucs.append(fast_auc(y, pk))
            preds.append(pk)
        aucs = np.array(aucs)                                       # (n_perm,) AUCs for each permutation of the pathway
        
        res = {"importance": float(auc0 - aucs.mean()),
               "se_perm": float(aucs.std(ddof=1) / np.sqrt(n_perm)) if n_perm > 1 else np.nan,
               "auc_perm_mean": float(aucs.mean())}
        
        # --- bootstrap confidence intervals ---
        if n_boot > 0:
            diffs = []
            for bi in boot_idx:
                a0 = fast_auc(y[bi], p0[bi])
                ak = np.mean([fast_auc(y[bi], pk[bi]) for pk in preds])
                diffs.append(a0 - ak)
            res["ci_low"], res["ci_high"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
        results[name] = res
        if return_predictions:
            preds_store[name] = np.stack(preds) # (n_perm, n, C) predictions for each permutation of the pathway
        if verbose:
            print(f"[{i + 1}/{len(names)}] {name[:50]:50s} I={res['importance']:+.4f} "
                  f"SE={res['se_perm']:.4f} ({time.time() - t1:.1f}s)")
    
    if return_predictions:
        return results, auc0, {"baseline": p0, "permuted": preds_store}
    return results, auc0
