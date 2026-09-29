
"""
KG-HTT v5: Hierarchical TabTransformer
=========================================================
- Legend -
B: samples in the batch (batch_size)
M: number of genes (n_features)
K: number of pathways
D: embedding dimension (embed_dim)
c: number of pathways in a chunk (chunk_size)
n_c: padded length of a chunk = size of its largest pathway
n_k: number of genes of pathway k
T: sequence length (n_c for local, K for global)
H: number of attention heads (num_heads)

- Input - 
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
    Transformer block:
    x1  = LayerNorm(x  + Dropout(MHA(x, x, x)))       # self-attention sub-layer
    out = LayerNorm(x1 + Dropout(FFN(x1)))            # position-wise feed-forward sub-layer
    
    Input and output have the same shape (B, T, D), so blocks can be stacked.
    The same class is used at both levels of the hierarchy:
      - local  level: T = genes of a pathway (padded to n_c inside a chunk), with a padding mask
      - global level: T = K pathways, no padding (padding_mask=None)
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        
        # --- Multi-Head Attention ---
        # Each head works in a subspace of dimension key_dim = embed_dim // num_heads.
        # dropout=rate is applied to the attention probabilities (after softmax) only during training. 
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=rate,
        )
        
        # --- Position-wise Feed-Forward Network: D -> ff_dim -> D ---
        # Two Dense Layers with a GELU activation in between. Dropout is applied after the first Dense layer
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        
        # --- LayerNorm ---
        # For each token, normalize its D values to mean 0 / variance 1,
        # then rescale with learned gamma (D,) and beta (D,). epsilon avoids division by zero.
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # --- Dropout on the output of each sub-layer, before the residual sum ---
        # In inference (training = False) is the identity
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False, return_attention=False, padding_mask=None):
        """
        Input x: (B, T, D) sequence of T tokens of size D.
        training: True during fit (dropout active), False at inference.
        padding_mask: (B, T), 1/True for real tokens, 0/False for padding. None = no padding. 
        Returns out (B, T, D) or (out, scores) if return_attention = True,
        where scores (B, H, T, D) are the attention weights of this block. 
        (scores[b, h, i, j] = how token i attends to token j in head h).
        """
        
        # --- Build the attention mask from the padding mask ---
        # Keras expects attention_mask of shape (B, T_query, T_key): True = "query i may attend to key j".
        # We mask the keys (columns): no token can attend to a padding position, so padding never enters 
        # the weighted sum of real tokens. Softmax assigns weight 0 to the padding columns. 
        att_mask = None
        if padding_mask is not None:
            key_mask = tf.cast(padding_mask, tf.bool)[:, None, :]          # (B, T) -> (B, 1, T)
            # repeat the same key mask for every query row: (B, 1, T) -> (B, T, T)
            att_mask = tf.broadcast_to(key_mask, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[1]]) # (B, T, T)


        # --- Self-attention: query = value = x (key defaults to value) ---
        if return_attention:
            attn, scores = self.att(x, x, training=training,
                                    return_attention_scores=True,
                                    attention_mask=att_mask)                    # attn (B, T, D), scores (B, H, T, T)
        else:
            attn = self.att(x, x, training=training, attention_mask=att_mask)   # attn (B, T, D)
            scores = None

        # --- Sub-layer 1: residual connection + LayerNorm --- 
        attn = self.drop1(attn, training=training)      # (B, T, D)
        x1 = self.norm1(x + attn)                       # (B, T, D)     x1 = LayerNorm(x + Dropout(MHA(x, x)))
        
        # --- Sub-layer 2: feed-forward + residual connection + LayerNorm ---
        ffn = self.ffn(x1, training=training)           # (B, T, D)
        ffn = self.drop2(ffn, training=training)        # (B, T, D)
        out = self.norm2(x1 + ffn)                      # (B, T, D)     out = LayerNorm(x1 + Dropout(FFN(x1)))
        return (out, scores) if return_attention else out


# ======================================================
# DUAL CONTINUOUS EMBEDDING
# ======================================================

class DualContinuousEmbedding(layers.Layer):
    """
    Turns each gene into a D-dimensional token, combining two "views" of the same number: 
        - bin embedding: a learned vector for (gene j, bin b)
        - value projection: e_val = x_j * w_j + b_j
        
    mode="sum":    e_bin (D) + e_val (D)                          -> (B, M, D)
    mode="concat": [e_bin (D) | e_val (D)] (2D) @ W (2D x D) + b  -> (B, M, D)
    
    per_feature_proj = True: each gene j has its own w_j, b_j       ()
    per_feature_poj = False: one w, b shared by all genes
    
    Fully vectorized: the M per-gene embedding tables are stored as one flat table 
    of (M * n_bins) rows, where row j * n_bins + b = (gene j, bin b).     
    """

    def __init__(self, n_features, n_bins, embed_dim, per_feature_proj=True, mode="sum"):
        super().__init__()
        assert mode in ("sum", "concat")
        self.n_features, self.n_bins, self.embed_dim = n_features, n_bins, embed_dim
        self.per_feature_proj, self.mode = per_feature_proj, mode

        # --- Bin Embedding Table (M * n_bins, D), block j = the n_bins rows of gene j ---
        self.bin_table = self.add_weight(
            shape=(n_features * n_bins, embed_dim),
            initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05), # default initializer of keras.layers.Embedding
            trainable=True, name="bin_table",
        )
        
        # offsets (M,) = [0, n_bins, 2*n_bins, ..., (M-1)*n_bins]: first row of each gene's block.
        self.offsets = tf.constant(np.arange(n_features, dtype=np.int32) * n_bins)  # (M,)

        # --- Linear projection of the continuous value: e_val = x_j * w_j + b_j
        lim = float(np.sqrt(6.0 / (1 + embed_dim))) # Glorot-uniform (default Dense initializer)
        w_shape = (n_features, embed_dim) if per_feature_proj else (1, embed_dim)
        self.value_w = self.add_weight(shape=w_shape,
                                       initializer=tf.keras.initializers.RandomUniform(-lim, lim),
                                       trainable=True, name="value_w")
        self.value_b = self.add_weight(shape=w_shape, initializer="zeros",
                                       trainable=True, name="value_b")

        if mode == "concat":
            # Shared projection R^{2D} -> R^D. With kernel = [I; I] and zero
            # bias this reproduces mode="sum" exactly (thus sum is a special case of concat)
            self.proj = layers.Dense(embed_dim, use_bias=True, name="concat_proj")

    def _combine(self, e_bin, e_val):
        if self.mode == "concat":
            return self.proj(tf.concat([e_bin, e_val], axis=-1))    # (B, n, 2D) -> Dense -> (B, n, D)
        return e_bin + e_val

    def call(self, x_cont, x_bins):
        """x_cont (B, M) float32, x_bins (B, M) int -> (B, M, D)"""
        # row index of (gene j, bin b) in the flat table = b + j * n_bins.
        # offsets[None, :] has shape (1, M) and is broadcast over the B rows of x_bins.
        # tf.gather(table, idx) returns table[idx] for every entry of idx: (M*n_bins, D)[(B, M)] -> (B, M, D)
        e_bin = tf.gather(self.bin_table, tf.cast(x_bins, tf.int32) + self.offsets[None, :])
        # x_cont[:, :, None] (B, M, 1) * value_w[None] (1, M, D) (or (1, 1, D) if shared) -> (B, M, D):
        e_val = x_cont[:, :, None] * self.value_w[None] + self.value_b[None]
        return self._combine(e_bin, e_val)

    def call_subset(self, x_cont_sub, x_bins_sub, gene_idx):
        """
        Same as call(), but only for the genes in gene_idx (1-D int tensor of length n),
        used by permutation importance to re-encode a single pathway without embedding all M genes.
        x_cont_sub, x_bins_sub: (B, n), already restricted to gene_idx.
        Returns (B, n, D). Identical to gather(call(...), gene_idx, axis=1).
        """
        gene_idx = tf.cast(gene_idx, tf.int32)
        e_bin = tf.gather(self.bin_table,
                          tf.cast(x_bins_sub, tf.int32) + tf.gather(self.offsets, gene_idx)[None, :])
        if self.per_feature_proj:
            w = tf.gather(self.value_w, gene_idx)   # (n, D) rows of the selected genes
            b = tf.gather(self.value_b, gene_idx)   # (n, D)
        else:
            w, b = self.value_w, self.value_b       # (1, D) shared by all genes
        e_val = x_cont_sub[:, :, None] * w[None] + b[None]  # (B, n, D)
        return self._combine(e_bin, e_val)


# =============================================================================
# ATTENTION POOLING (used at both levels: genes -> pathways, pathways -> head)
# =============================================================================

class AttentionPooling(layers.Layer):
    """
    Learned weighted average of a sequence of T vectors, with n_queries learned queries.
     
    s^(h)_t = <q_h, W h_t> / sqrt(D)        w^(h) = softmax_t(s^(h))
    z_h     = sum_t w^(h)_t * h_t           z = concat_h(z_h)   -> (N, n_queries * D)
     
    Each query q_h is a learned "question" (a direction in R^D). The weights w^(h)_t depend on the
    content of h_t, so they change from sample to sample, unlike a fixed mean.
    The output is a weighted average of the h_t themselves (no value projection), so it stays in
    the same space as the mean it replaces.
     
    Used at both levels of the hierarchy:
    - gene level (T = genes of a pathway, n_queries = 1): replaces the masked mean,
    w_t = beta_i is the weight of gene i inside that pathway for that sample.
    One query only, because the output must stay D-dimensional (it is a token of
    the global transformer).
    - pathway level (T = K pathways): replaces the mean before the head,
    w_t = alpha_k is the weight of pathway k for that sample. 
    Several queries are allowed and widen the vector reaching the head.
    
    With q = 0 the softmax is uniform and this reduces exactly to the (masked) mean, so
    mean pooling is a special case.
    """
    
    def __init__(self, embed_dim, n_queries=1):
        super().__init__()
        self.embed_dim, self.n_queries = embed_dim, n_queries
        # key projection W (D x D), no bias: k_t = W h_t
        self.key_proj = layers.Dense(embed_dim, use_bias=False, name="pool_key")
        # learned queries (n_queries, D), one row per query
        self.queries = self.add_weight(
            shape=(n_queries, embed_dim), initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True, name="pool_queries") # training starts near-uniform weights, that is the mean

    def call(self, h, pad_mask=None, return_weights=False):
        """
        h: (N, T, D), pad_mask: (N, T) or None -> (N, n_queries * D) [, w (N, n_queries, T)]
        N = number of sequences pooled at once (B*c pathways at gene level, B samples at pathway level)
        """
        
        keys = self.key_proj(h)     # (N, T, D)
        # einsum: every letter names an axis and the letters missing from the output are summed over.
        # "hd,ntd->nht": scores[n, h, t] = sum_d queries[h, d] * keys[n, t, d] = <q_h, k_t>
        # scaled by sqrt(D) as in standard attention, to keep the softmax from saturating.
        scores = tf.einsum("hd,ntd->nht", self.queries, keys) / tf.sqrt(float(self.embed_dim)) # (N, n_queries, T)
        
        if pad_mask is not None:
        # same trick as the attention mask in MultiHeadAttention, written by hand:
        # pad_mask[:, None, :] (N, 1, T) is broadcast over the n_queries axis; padding positions
        # get -1e9 added to their score, so exp(score) = 0 and their softmax weight is exactly 0.
            scores += (1.0 - pad_mask[:, None, :]) * tf.constant(-1e9, scores.dtype) # (N, n_queries, T)
        w = tf.nn.softmax(scores, axis=-1)      # (N, n_queries, T), sums to 1 over T
        # "nht,ntd->nhd": z[n, h, :] = sum_t w[n, h, t] * h[n, t, :]  (weighted average of the tokens)
        z = tf.einsum("nht,ntd->nhd", w, h)     # (N, n_queries, D)
        # concatenate the n_queries outputs of each sequence
        z = tf.reshape(z, [tf.shape(h)[0], self.n_queries * self.embed_dim]) # (N, n_queries, D) -> (N, n_queries * D)
        return (z, w) if return_weights else z


# ======================================================
# HIERARCHICAL TABTRANSFORMER
# ======================================================

class HierarchicalTabTransformer(Model):
    """
    Genes -> embedding -> local transformer (per pathway, weights shared by all pathways)
        -> gene pooling (masked mean or attention, weights beta) -> pathway vectors z (B, K, D)
        -> global transformer (over the K pathways)
        -> pathway pooling (mean, attention or multi-query attention, weights alpha) -> head.
    Pathways are sorted by size and processed in chunks of chunk_size pathways; each chunk is
    padded only to the size n_c of its largest pathway, and a padding mask excludes the padding.
    
    The two poolings are independent: either, both or neither can be attention-weighted. 
    """

    def __init__(self, pathway_map, n_features, n_bins=6, embed_dim=64,
                 num_heads=4, ff_dim=128, local_layers=2, global_layers=2,
                 dropout=0.1, task="binary", n_classes=None,
                 per_feature_proj=True, mode="sum", chunk_size=32,
                 gene_pooling="mean", pathway_pooling="mean", n_queries=4,
                 pooling=None):
        """
        patway_map: dict {pathwayname: list of gene column indices in 0..M-1} (genes can belong to several pathways)
        n_features: M, number of gene columns of x_cont / x_bins
        chunk_size: number of pathways processed together by the local trasformer
        gene_pooling: "mean" or "attention"; genes -> one vector per pathway
        pathway_pooling: "mean", "attention" or "multi_attention"; pathways -> head input
        n_queries: number of queries for "multi_attention" (ignored otherwise)
        """
        super().__init__()
        if pooling is not None:
            pathway_pooling = pooling       # backward compatibility with the old name
        assert gene_pooling in ("mean", "attention")
        assert pathway_pooling in ("mean", "attention", "multi_attention")
        self.pathway_map = pathway_map
        self.task, self.mode = task, mode
        self.gene_pooling, self.pathway_pooling = gene_pooling, pathway_pooling
        # number of pathway-level queries: "attention" -> 1, "multi_attention" -> n_queries, "mean" -> 0 (no pooling layer)
        self.n_queries = 1 if pathway_pooling == "attention" else (n_queries if pathway_pooling == "multi_attention" else 0)
        self.chunk_size, self.embed_dim = chunk_size, embed_dim
        self.n_features = n_features

        # ============================================================
        # Pathway index matrices and padding masks, bucketed by size
        # ============================================================
        # Pathways are sorted by size and chunked in that order. Each chunk is padded
        # only to the size of its largest pathway, to reduce the local attention cost. 
        # Pathway vectors are returned in the original order of pathway_map
        
        self.pathway_names = list(pathway_map.keys())  # original ordering
        self.n_pathways = len(self.pathway_names)      # K
        self.n_max = max(len(v) for v in pathway_map.values()) # largest pathway
        # padding slot: _embed_with_pad appends an all-zero token at position M of the (B, M+1, D) embedding,
        # so the index M (= n_features) in a pathway index matrix means "padding"
        self.pad_index = n_features 

        sizes = np.array([len(pathway_map[n]) for n in self.pathway_names])    # (K,) genes per pathway
        # argsort returns the indices that would sort sizes: sorted_order[i] = original index of the
        # i-th smallest pathway. kind="stable": pathways of equal size keep their original order
        self.sorted_order = np.argsort(sizes, kind="stable")    # sorted position -> original index
        # argsort of a permutation = its inverse: inverse_order[k] = position of pathway k in the sorted order.
        # Used at the end of _pathway_vectors to put the pathway vectors back in the original order.
        self.inverse_order = tf.constant(np.argsort(self.sorted_order).astype(np.int32))  # original -> sorted position
        
        
        self.chunks = []        # list of (orig_indices, idx (c, n_c), mask (c, n_c))
        for start in range(0, self.n_pathways, chunk_size):     # start = 0, chunk_size, 2*chunk_size, ...
            orig = self.sorted_order[start:start + chunk_size]  # original indices of the c pathways of this chunk
            n_c = int(sizes[orig].max())    # padded length = largest pathway of the chunk
            
            # start from "all padding": every index = pad_index and every mask entry = 0
            idx = np.full((len(orig), n_c), self.pad_index, dtype=np.int32)     # (c, n_c)
            msk = np.zeros((len(orig), n_c), dtype=np.float32)                  # (c, n_c)
            
            for j, oi in enumerate(orig): # j = row in the chunk, oi = original pathway index
                g = np.asarray(pathway_map[self.pathway_names[oi]], dtype=np.int32)     # gene indices of the pathway
                # overwrite the first len(g) positions of each row with the real genes and set their mask to 1:
                idx[j, :len(g)], msk[j, :len(g)] = g, 1.0
            # save a tuple (original indices of the pathways, idx matrix, msk matrix)
            self.chunks.append((orig, tf.constant(idx), tf.constant(msk)))
        
        # ================================
        # Layers
        # ================================
        self.embedding_layer = DualContinuousEmbedding(
            n_features, n_bins, embed_dim, per_feature_proj, mode)
        # the same local blocks are applied to every pathway (sahred weights), thus the number of parameters does not grow with K
        self.local_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                             for _ in range(local_layers)]
        self.global_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                              for _ in range(global_layers)]
        self.gene_pool = AttentionPooling(embed_dim, 1) if gene_pooling == "attention" else None
        self.pool = None if pathway_pooling == "mean" else AttentionPooling(embed_dim, self.n_queries)

        # output activation and size depending on the task (binary, multiclass, regression) 
        if task == "binary":
            out_units, out_act = 1, "sigmoid"
        elif task == "multiclass":
            assert n_classes is not None
            out_units, out_act = n_classes, "softmax"
        else:
            out_units, out_act = 1, "linear"
        
        # MLP head (128 -> 64 -> output)
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
        x = self.embedding_layer(x_cont, x_bins)                                # (B, M, D)
        # one extra all-zero token per sample
        zeros = tf.zeros([tf.shape(x)[0], 1, self.embed_dim], dtype=x.dtype)    # (B, 1, D)
        # append at position M, therefore gathering pad_index M returns the zero vector
        return tf.concat([x, zeros], axis=1)                                    # (B, M+1, D)

    @staticmethod # no self
    def _masked_mean(h, mask):
        """h (N, T, D), mask (N, T) -> (N, D). Mean over the real positions only"""
        m = mask[:, :, None]    # (N, T, 1), broadcast over the D dims
        # numerator: sum of the real tokens only since padding is multiplied by 0
        # denominator: number of real tokens n_k
        return tf.reduce_sum(h * m, axis=1) / tf.reduce_sum(m, axis=1)


    def _run_local(self, seq, mask, training=False, return_attention=False):
        """
        Local transformer (within pathway) + gene pooling. Applied to N = B * c pathway sequences at once
        seq (N, T, D), mask (N, T) -> pooled (N, D)
        [, list of attention scores, gene pooling weights beta (N, T) or None]
        """
        h, per_layer = seq, []
        for blk in self.local_blocks:
            if return_attention:
                h, s = blk(h, training=training, return_attention=True, padding_mask=mask)
                per_layer.append(s)     # scores of this layer: (N, H, T, T)
            else:
                h = blk(h, training=training, padding_mask=mask)
        
        if self.gene_pool is None:
            pooled, beta = self._masked_mean(h, mask), None
        elif return_attention:
            pooled, w = self.gene_pool(h, pad_mask=mask, return_weights=True)   # w (N, 1, T)
            beta = w[:, 0, :]   # only one query at gene level, drop that axis -> (N, T)
        else:
            pooled, beta = self.gene_pool(h, pad_mask=mask), None
        return (pooled, per_layer, beta) if return_attention else pooled


    def _pathway_vectors(self, x, training=False, return_attention=False):
        """
        x (B, M+1, D) -> z (B, K, D). Chunked over pathways.
        If return_attention, also returns {pathway: [scores per layer (B, H, n_k, n_k)]}
        and {pathway: beta (B, n_k)} with the gene pooling weights (empty for mean pooling).
        """
        B = tf.shape(x)[0]
        D = self.embed_dim
        all_vecs, local_scores, gene_weights = [], {}, {}
        for orig, pw_idx, pw_mask in self.chunks:                # size-bucketed chunks
            c, n_c = pw_idx.shape                                # pathways in this chubk, padded length
            # gather the gene tokens of each pathway of the chunk; pad index picks the zero row
            xp = tf.gather(x, pw_idx, axis=1)                    # (B, M+1, D)[(c, n_c)] -> (B, c, n_c, D)
            # flatten samples and pathways into one batch axis, to call the local transformer once
            xp_flat = tf.reshape(xp, [B * c, n_c, D])            # row = b*c + j (pathway j of sample b)
            # the mask is the same for every sample, stacking the (c, n_c) block B times
            mask_flat = tf.tile(pw_mask, [B, 1])                 # row = b*c + j (pathway j of sample b)


            if return_attention:
                pooled, per_layer, beta = self._run_local(xp_flat, mask_flat, training, True)
                for j, oi in enumerate(orig):
                    name = self.pathway_names[oi]
                    n_k = len(self.pathway_map[name]) # real number of genes
                    # s[j::c] = rows j, j+c, j+2c, ... = pathway j for samples 0..B-1 -> (B, H, n_c, n_c)
                    # [:, :, :n_k, :n_k] cuts away the padded rows and columns
                    local_scores[name] = [s[j::c][:, :, :n_k, :n_k] for s in per_layer]   
                    # same as above for the gene-pooling weights beta 
                    if beta is not None:
                        gene_weights[name] = beta[j::c][:, :n_k]           # (B, n_k)
            else:
                pooled = self._run_local(xp_flat, mask_flat, training)     # (B*c, D), one call to the local transformer
            all_vecs.append(tf.reshape(pooled, [B, c, D]))                 # undo the above flattening: (B, c, D)

        z_sorted = tf.concat(all_vecs, axis=1)                   # (B, K, D) in sorted order
        # inverse_order[k] = position of pathway k in the sorted order -> restores original patwhay_map order
        z = tf.gather(z_sorted, self.inverse_order, axis=1)      # back to original order
        return (z, local_scores, gene_weights) if return_attention else z

    def _global_and_head(self, z, training=False, return_attention=False):
        """
        z (B, K, D) -> predictions. 
        Global transformer over the K pathway tokens, pathway pooling and then MLP head. 
        No padding mask here.
        """
        
        h, global_scores = z, []
        for blk in self.global_blocks:
            if return_attention:
                h, s = blk(h, training=training, return_attention=True)     # s (B, H, K, K)
                global_scores.append(s)
            else:
                h = blk(h, training=training)
        if self.pool is None:
            pooled, alpha = tf.reduce_mean(h, axis=1), None     # plain mean: (B, K, D) -> (B, D)
        elif return_attention:
            pooled, alpha = self.pool(h, return_weights=True)   # alpha (B, n_queries, K)
        else:
            pooled, alpha = self.pool(h), None
        out = self.head(pooled, training=training)      # (B, out_units)
        return (out, global_scores, alpha) if return_attention else out

    # --------------------------------------------------
    # forward
    # --------------------------------------------------
    def call(self, inputs, training=False):
        x_cont, x_bins = inputs                              # unpack the list of two (B, M) tensors
        x = self._embed_with_pad(x_cont, x_bins)             # (B, M+1, D)
        z = self._pathway_vectors(x, training=training)      # (B, K, D)
        return self._global_and_head(z, training=training)   # (B, out_units)

    def forward_with_attention(self, inputs):
        """
        Same forwars as call() in inference mode
        - Returns -
        out: predictions - (B, out_units)
        local_scores: attention gene-gene within each pathway - dict {pathway: [(B, H, n_k, n_k)]}
        global_scores: attention pathway-pathway - list (B, H, K, K)
        alpha: pathways weights - (B, n_queries, K) or None
        beta: genes weights - dict {pathway: (B, n_k)} - empty dict for mean gene pooling
        """
        x_cont, x_bins = inputs
        x = self._embed_with_pad(x_cont, x_bins)
        z, local_scores, gene_weights = self._pathway_vectors(x, training=False, return_attention=True)
        out, global_scores, alpha = self._global_and_head(z, training=False, return_attention=True)
        return out, local_scores, global_scores, alpha, gene_weights

    # ----------------------------------------------------------
    # cached-pathway inference (used by permutation importance)
    # ----------------------------------------------------------
    # Rationale: permuting one pathway changes only one of the K pathway vectors,
    # so the expensive part (embedding and local transformer) can be done once and
    # cached. Per permutation only the target pathway is re-encoded, its column is swapped into
    # the cached z, and the global transformer + head are re-run.
    
    @tf.function(reduce_retracing=True)
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
        gene_idx = tf.constant(np.asarray(self.pathway_map[name], dtype=np.int32)) # (n_k, )
        xc = tf.gather(x_cont, gene_idx, axis=1)                   # (B, n_k) columns of this pathway
        xb = tf.gather(x_bins, gene_idx, axis=1)                   # (B, n_k)
        e = self.embedding_layer.call_subset(xc, xb, gene_idx)     # (B, n_k, D)
        
        if perm_idx is not None:
            # the embeddings are shuffled, is equivalent to shuffling the raw data
            perm_idx = tf.cast(perm_idx, tf.int32)
            if gene_pos is None:
                # full pathway permuted jointly
                e = tf.gather(e, perm_idx, axis=0)
            else:
                # permute a single gene: take its column, shuffle it and put it back
                col = tf.gather(e[:, gene_pos:gene_pos + 1, :], perm_idx, axis=0)
                e = tf.concat([e[:, :gene_pos, :], col, e[:, gene_pos + 1:, :]], axis=1)
        # all ones mask since every position is "real"
        mask = tf.ones([tf.shape(e)[0], tf.shape(e)[1]], dtype=tf.float32) # (B, n_k)
        return self._run_local(e, mask, training=False) # (B, D)

    @tf.function(reduce_retracing=True)
    def predict_from_pathway_vectors(self, z):
        """z (B, K, D) -> predictions, no dropout"""
        return self._global_and_head(z, training=False)

    def forward_with_pathway_permutation(self, inputs, target_pathway, perm_idx, z_cache=None):
        """Full-batch convenience wrapper: permute only `target_pathway`."""
        x_cont, x_bins = inputs
        t = self.pathway_names.index(target_pathway)    # list.index: name -> position
        # reuse cached pathway vectors if available, otherwise compute them
        z = self.compute_pathway_vectors(x_cont, x_bins) if z_cache is None else z_cache
        return self.predict_from_pathway_vectors(self._replace_column(
            z, t, self.encode_pathway_subset(x_cont, x_bins, t, perm_idx)))

    def forward_with_gene_permutation(self, inputs, target_pathway, gene_idx, perm_idx, z_cache=None):
        """Permute one gene (global index gene_idx) only inside `target_pathway`."""
        x_cont, x_bins = inputs
        t = self.pathway_names.index(target_pathway)
        pos = list(self.pathway_map[target_pathway]).index(gene_idx) # global gene index -> position in the pathway
        # reuse cached pathway vectors if available, otherwise compute them
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
    
    Uses the Mann-Whitney identity AUC = P(score(positive) > score(negative)), computed
    from ranks instead of ROC points. 
    
    AUC = (R - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    with R = sum of the rank of the positives
    
    rankdata gives average ranks to ties, same as sklearn.
    """
    from scipy.stats import rankdata
    scores = np.asarray(scores)
    # binary case: build the two columns [P(class 0), P(class 1)] so the code below is uniform
    if scores.ndim == 1 or scores.shape[1] == 1:
        scores = np.column_stack([1 - scores.ravel(), scores.ravel()])
    n, C = scores.shape
    ranks = rankdata(scores, axis=0)  # (n, C) ranks within each column (ties averaged)
    aucs = []
    for c in range(C):
        pos = (y == c)
        n_pos, n_neg = pos.sum(), n - pos.sum()
        if n_pos == 0 or n_neg == 0: # if a class is absent, skip it, AUC is undefined
            continue
        aucs.append((ranks[pos, c].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs))     # average over the classes


# ======================================================
# PERMUTATION IMPORTANCE WITH CACHED PATHWAY VECTORS
# ======================================================

def _batched(fn, n, batch_size):
    """ Apply fn(start, end) over slices of n rows and stack the results (fn is a function)"""
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
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(boot_seed)
    # bootstrap resamples: n indices drawn with replacement
    boot_idx = [brng.integers(0, n, n) for _ in range(n_boot)]

    t0 = time.time()
    z = _batched(lambda s, e: model.compute_pathway_vectors(X_cont[s:e], X_bins[s:e]).numpy(),
                 n, batch_size)                                                    # (n, K, D)
    p0 = _batched(lambda s, e: model.predict_from_pathway_vectors(tf.constant(z[s:e])).numpy(),
                  n, batch_size)                                                    # (n, C)
    auc0 = fast_auc(y, p0)
    if verbose:
        print(f"cached pathway vectors {z.shape} in {time.time() - t0:.1f}s, baseline AUC {auc0:.4f}")

    perms = [rng.permutation(n) for _ in range(n_perm)]    # same permutations for every pathway
    names = pathways if pathways is not None else model.pathway_names   # optional subset of pathways
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
            z_mod[:, t, :] = z[perm, t, :]      # only column t is shuffled across samples
            pk = _batched(lambda s, e: model.predict_from_pathway_vectors(tf.constant(z_mod[s:e])).numpy(),
                          n, batch_size)
            aucs.append(fast_auc(y, pk))
            preds.append(pk)
        aucs = np.array(aucs)
        res = {"importance": float(auc0 - aucs.mean()), # AUC lost by destroying P 
               "se_perm": float(aucs.std(ddof=1) / np.sqrt(n_perm)) if n_perm > 1 else np.nan, # Monte Carlo se
               "auc_perm_mean": float(aucs.mean())}
        if n_boot > 0:
            # percentile bootstrap: recompute the same difference on resempled test sets
            diffs = []
            for bi in boot_idx:
                a0 = fast_auc(y[bi], p0[bi])
                ak = np.mean([fast_auc(y[bi], pk[bi]) for pk in preds])
                diffs.append(a0 - ak)
            res["ci_low"], res["ci_high"] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]
        results[name] = res
        if return_predictions:
            preds_store[name] = np.stack(preds)  # (n_perm, n, C)
        if verbose:
            print(f"[{i + 1}/{len(names)}] {name[:50]:50s} I={res['importance']:+.4f} "
                  f"SE={res['se_perm']:.4f} ({time.time() - t1:.1f}s)")
    if return_predictions:
        return results, auc0, {"baseline": p0, "permuted": preds_store}
    return results, auc0
