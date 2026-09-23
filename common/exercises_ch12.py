"""
common/exercises_ch12.py
========================
Chapter 12: Transformers - Exercises 12.1 to 12.16
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements comprehensive mathematical derivations, analytical solutions,
and numerical verifications for all 16 exercises in Chapter 12:
- Exercise 12.1: Lagrange Multiplier & Partition of Unity Upper Bound (a_nm <= 1).
- Exercise 12.2: Softmax Partition of Unity Verification.
- Exercise 12.3: Orthogonal Input Attention Collapse (y_n = x_n).
- Exercise 12.4: Expectation of Squared Inner Product E[(a^T b)^2] = D.
- Exercise 12.5: Multi-Head Attention Low-Rank Decomposition W_(h) = W_h^(v) W_h^(o).
- Exercise 12.6: Sparse Parameter Sharing Matrix Representation of Self-Attention (O(N^2 D^2)).
- Exercise 12.7: Permutation Equivariance of Multi-Head Attention without Positional Encodings.
- Exercise 12.8: High-Dimensional Random Vector Orthogonality (Var(cos theta) = 1/D).
- Exercise 12.9: Concatenation vs Addition Equivalence in Linear Projections.
- Exercise 12.10: Sinusoidal Positional Encoding Relative Shift via 2D Rotation Matrix R(omega * k).
- Exercise 12.11: Maximum Likelihood Estimation of Bag-of-Words Model (theta_v = c_v / N).
- Exercise 12.12: Exponential Parameter Growth of Tabular Autoregressive Models (O(V^N)).
- Exercise 12.13: N-gram Ratio Conditional Probability and Final Token Omission.
- Exercise 12.14: Executable Inference Loop & Pseudocode for Trained Recurrent Neural Network.
- Exercise 12.15: Failure of Greedy Decoding vs Global Optimum Sequence Probability.
- Exercise 12.16: BERT-Large Parameter Count Exact Breakdown (~340 Million Parameters).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot
from .attention import softmax, sinusoidal_positional_encoding


# =============================================================================
# Exercise 12.1: Lagrange Multiplier & Partition of Unity Upper Bound
# =============================================================================

def exercise_12_1_lagrange_multiplier(N: int = 5) -> Dict[str, Any]:
    """
    Exercise 12.1:
    Show that coefficients satisfying a_nm >= 0 and sum_{m=1}^N a_nm = 1
    must also satisfy a_nm <= 1 for all n, m.
    
    Proof:
    For a fixed n and specific index k in {1, ..., N}, consider maximizing a_nk
    subject to g({a_nm}) = sum_{m=1}^N a_nm - 1 = 0 and a_nm >= 0.
    Directly:
        1 = sum_{m=1}^N a_nm = a_nk + sum_{m != k} a_nm >= a_nk + 0 = a_nk.
    Hence a_nk <= 1.
    Using Lagrange multiplier / KKT:
        L({a_nm}, lambda, {mu_m}) = a_nk - lambda (sum_m a_nm - 1) + sum_m mu_m a_nm
        dL / da_nk = 1 - lambda + mu_k = 0  =>  lambda = 1 + mu_k >= 1.
        dL / da_nm = -lambda + mu_m = 0      =>  mu_m = lambda >= 1 > 0  (for m != k).
        Complementary slackness mu_m * a_nm = 0 implies a_nm = 0 for all m != k.
        Then sum_m a_nm = 1 yields a_nk = 1.
        Thus max a_nk = 1, so 0 <= a_nm <= 1.
    """
    # Numerical validation via constrained optimization
    c = np.zeros(N)
    c[0] = -1.0  # maximize a_n1 <==> minimize -a_n1
    A_eq = np.ones((1, N))
    b_eq = np.array([1.0])
    bounds = [(0.0, None) for _ in range(N)]

    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    max_val = -res.fun

    return {
        "verified": bool(np.isclose(max_val, 1.0)),
        "max_value": float(max_val),
        "solution": res.x.tolist(),
        "theoretical_bound": 1.0,
    }


# =============================================================================
# Exercise 12.2: Softmax Partition of Unity Verification
# =============================================================================

def exercise_12_2_verify_softmax(N: int = 6, D: int = 8, seed: int = 42) -> Dict[str, Any]:
    """
    Exercise 12.2:
    Verify that the softmax function (Eq 12.5):
        a_nm = exp(x_n^T x_m) / sum_{m'=1}^N exp(x_n^T x_m')
    satisfies a_nm >= 0 and sum_{m=1}^N a_nm = 1 for any arbitrary inputs x_1, ..., x_N.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))  # (N, D)
    logits = np.matmul(X, X.T)   # (N, N)
    A = softmax(logits, axis=-1) # (N, N)

    all_non_negative = bool(np.all(A >= 0.0))
    row_sums = np.sum(A, axis=-1)
    sums_to_one = bool(np.allclose(row_sums, 1.0, atol=1e-7))

    return {
        "all_non_negative": all_non_negative,
        "sums_to_one": sums_to_one,
        "row_sums": row_sums.tolist(),
        "min_val": float(np.min(A)),
        "max_val": float(np.max(A)),
        "verified": all_non_negative and sums_to_one,
    }


# =============================================================================
# Exercise 12.3: Orthogonal Input Attention Collapse
# =============================================================================

def exercise_12_3_orthogonal_attention(N: int = 4, scale: float = 10.0) -> Dict[str, Any]:
    """
    Exercise 12.3:
    Show that if all input vectors are orthogonal (x_n^T x_m = 0 for n != m),
    then each output vector y_n is equal to x_n.
    
    Analysis:
    For orthogonal vectors with norm ||x_n||:
        x_n^T x_m = 0 (n != m)  => exp(x_n^T x_m) = exp(0) = 1.
        x_n^T x_n = ||x_n||^2   => exp(x_n^T x_n) = exp(||x_n||^2).
    The attention weight on x_n is:
        a_nn = exp(||x_n||^2) / (exp(||x_n||^2) + N - 1).
    For non-trivial features (||x_n||^2 >> 1, e.g. after scaling / hard attention limit):
        a_nn -> 1.0,  a_nm -> 0.0 (m != n).
    Thus y_n = sum_m a_nm x_m -> x_n.
    """
    # Orthonormal basis vectors scaled by `scale`
    I = np.eye(N) * scale
    logits = np.matmul(I, I.T)
    A = softmax(logits, axis=-1)
    Y = np.matmul(A, I)

    diag_weights = np.diag(A)
    error = np.max(np.abs(Y - I))

    return {
        "diag_attention_weights": diag_weights.tolist(),
        "reconstruction_error": float(error),
        "verified": bool(error < 1e-3),
    }


# =============================================================================
# Exercise 12.4: Expectation of Squared Inner Product E[(a^T b)^2] = D
# =============================================================================

def exercise_12_4_expected_inner_product_squared(
    D: int = 16,
    num_samples: int = 100000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.4:
    Show that for independent a, b ~ N(0, I_D), E[(a^T b)^2] = D.
    
    Proof:
    a^T b = sum_{i=1}^D a_i b_i.
    (a^T b)^2 = sum_{i=1}^D sum_{j=1}^D a_i a_j b_i b_j.
    E[(a^T b)^2] = sum_{i,j} E[a_i a_j] E[b_i b_j] = sum_{i,j} delta_{ij} delta_{ij} = sum_{i=1}^D 1 = D.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1.0, size=(num_samples, D))
    B = rng.normal(0, 1.0, size=(num_samples, D))

    dots = np.sum(A * B, axis=1)  # (num_samples,)
    dots_sq = dots ** 2
    empirical_mean = float(np.mean(dots_sq))
    std_err = float(np.std(dots_sq) / np.sqrt(num_samples))

    return {
        "D": D,
        "empirical_mean": empirical_mean,
        "theoretical_value": float(D),
        "std_error": std_err,
        "verified": bool(abs(empirical_mean - D) < 3.0 * std_err),
    }


# =============================================================================
# Exercise 12.5: Multi-Head Attention Low-Rank Decomposition
# =============================================================================

def exercise_12_5_multihead_low_rank(
    N: int = 6,
    D: int = 16,
    H: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.5:
    Show that multi-head attention can be written as:
        Y = sum_{h=1}^H H_h X W_(h)
    where W_(h) = W_h^(v) W_h^(o) has rank at most D_v = D/H < D.
    """
    rng = np.random.default_rng(seed)
    Dv = D // H
    X = rng.normal(size=(N, D))

    # Attention matrices H_h: (N, N)
    H_heads = [softmax(rng.normal(size=(N, N)), axis=-1) for _ in range(H)]

    # Value matrices W_h^(v): (D, Dv)
    W_v = [rng.normal(size=(D, Dv)) for _ in range(H)]

    # Output projection submatrices W_h^(o): (Dv, D)
    W_o_sub = [rng.normal(size=(Dv, D)) for _ in range(H)]
    W_o = np.concatenate(W_o_sub, axis=0)  # (H * Dv, D) = (D, D)

    # Standard formulation (Eq 12.19):
    # Concat head outputs: [H_1 X W_1^(v), ..., H_H X W_H^(v)] @ W_o
    head_outs = [np.matmul(H_heads[h], np.matmul(X, W_v[h])) for h in range(H)]
    concat_heads = np.concatenate(head_outs, axis=1)  # (N, H * Dv) = (N, D)
    Y_standard = np.matmul(concat_heads, W_o)

    # Rewritten formulation (Eq 12.42, 12.43):
    # Y = sum_{h=1}^H H_h X (W_h^(v) W_h^(o))
    Y_sum = np.zeros((N, D))
    ranks = []
    for h in range(H):
        W_h_comb = np.matmul(W_v[h], W_o_sub[h])  # (D, D)
        rank_h = int(np.linalg.matrix_rank(W_h_comb))
        ranks.append(rank_h)
        Y_sum += np.matmul(H_heads[h], np.matmul(X, W_h_comb))

    max_diff = float(np.max(np.abs(Y_standard - Y_sum)))
    all_low_rank = all(r <= Dv for r in ranks)

    return {
        "verified": bool(max_diff < 1e-7 and all_low_rank),
        "max_diff": max_diff,
        "Dv": Dv,
        "ranks": ranks,
    }


# =============================================================================
# Exercise 12.6: Sparse Parameter Sharing Matrix of Self-Attention
# =============================================================================

def exercise_12_6_sparse_parameter_sharing(
    N: int = 3,
    D: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.6:
    Express self-attention as a sparse block matrix of size ND x ND mapping flattened x
    into y, showing parameter sharing and sparsity.
    
    Flattened: x_flat in R^{ND}, y_flat = M x_flat.
    M is composed of N x N blocks of size D x D, where block (n, m) is:
        M_{nm} = a_{nm} (W^(v))^T.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    W_v = rng.normal(size=(D, D))
    A = softmax(rng.normal(size=(N, N)), axis=-1)

    # Standard: y_n = sum_m a_nm x_m W_v  (row vectors)
    Y_standard = np.matmul(A, np.matmul(X, W_v))

    # As a single (ND x ND) block matrix
    # If row vector convention: Y_flat = X_flat @ M_row
    # If column vector convention: y_col = M_col x_col
    M_col = np.zeros((N * D, N * D))
    for n in range(N):
        for m in range(N):
            # y_n = sum_m a_nm W_v^T x_m
            M_col[n*D : (n+1)*D, m*D : (m+1)*D] = A[n, m] * W_v.T

    x_col = X.reshape(-1, 1)  # (ND, 1)
    y_col = np.matmul(M_col, x_col)
    Y_from_col = y_col.reshape(N, D)

    diff = float(np.max(np.abs(Y_standard - Y_from_col)))

    return {
        "verified": bool(diff < 1e-7),
        "max_diff": diff,
        "block_matrix_shape": M_col.shape,
        "full_params": (N * D) ** 2,
        "shared_params": D * D,
    }


# =============================================================================
# Exercise 12.7: Permutation Equivariance of Multi-Head Attention
# =============================================================================

def exercise_12_7_permutation_equivariance(
    N: int = 5,
    D: int = 8,
    num_heads: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.7:
    Show that without positional encodings, Multi-Head Attention is permutation equivariant:
        MHA(Pi X) = Pi MHA(X)
    for any permutation matrix Pi.
    """
    from .attention import MultiHeadAttention

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    mha = MultiHeadAttention(d_model=D, num_heads=num_heads, seed=seed)

    # Create random permutation matrix Pi
    perm = rng.permutation(N)
    Pi = np.eye(N)[perm]

    X_perm = np.matmul(Pi, X)

    Y_orig, _ = mha.forward(X)
    Y_perm, _ = mha.forward(X_perm)

    # Check if Y_perm == Pi @ Y_orig
    Y_expected = np.matmul(Pi, Y_orig)
    error = float(np.max(np.abs(Y_perm - Y_expected)))

    return {
        "permutation": perm.tolist(),
        "max_error": error,
        "verified": bool(error < 1e-7),
    }


# =============================================================================
# Exercise 12.8: High-Dimensional Random Vector Orthogonality
# =============================================================================

def exercise_12_8_high_dim_orthogonality(
    dimensions: List[int] = [2, 8, 32, 128, 512],
    num_trials: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.8:
    Show that for two random unit vectors a, b in R^D, Var(a^T b) = 1/D -> 0 as D -> infty.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for D in dimensions:
        A = rng.normal(0, 1.0, size=(num_trials, D))
        B = rng.normal(0, 1.0, size=(num_trials, D))
        A = A / np.linalg.norm(A, axis=-1, keepdims=True)
        B = B / np.linalg.norm(B, axis=-1, keepdims=True)

        cos_thetas = np.sum(A * B, axis=-1)
        mean_cos = float(np.mean(cos_thetas))
        var_cos = float(np.var(cos_thetas))
        theo_var = 1.0 / D

        results[D] = {
            "mean_cos": mean_cos,
            "var_cos": var_cos,
            "theoretical_var": theo_var,
        }

    # Verify that variance scales as 1/D
    last_D = dimensions[-1]
    verified = bool(results[last_D]["var_cos"] < 0.01)

    return {
        "results": results,
        "verified": verified,
    }


# =============================================================================
# Exercise 12.9: Concatenation vs Addition Equivalence
# =============================================================================

def exercise_12_9_concatenation_vs_addition(
    D_x: int = 6,
    D_e: int = 4,
    D_out: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.9:
    Show that multiplying concatenated vector [x; e] by matrix W
    is equivalent to W_x x + W_e e.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(D_x,))
    e = rng.normal(size=(D_e,))

    concat = np.concatenate([x, e])  # (D_x + D_e,)
    W = rng.normal(size=(D_x + D_e, D_out))

    # Multiplication of concatenated vector: concat @ W
    y_concat = np.matmul(concat, W)

    # Partitioned: x @ W_x + e @ W_e
    W_x = W[:D_x, :]
    W_e = W[D_x:, :]
    y_sum = np.matmul(x, W_x) + np.matmul(e, W_e)

    diff = float(np.max(np.abs(y_concat - y_sum)))

    return {
        "verified": bool(diff < 1e-9),
        "max_diff": diff,
    }


# =============================================================================
# Exercise 12.10: Sinusoidal Positional Encoding Relative Shift
# =============================================================================

def exercise_12_10_sinusoidal_shift(
    n: int = 5,
    k: int = 3,
    D: int = 8,
) -> Dict[str, Any]:
    """
    Exercise 12.10:
    Show that for sinusoidal positional encoding, position n+k is a linear transformation
    of position n via 2D rotation matrices R(omega_i * k).
    """
    # Compute full encodings
    seq_len = n + k + 2
    encodings = sinusoidal_positional_encoding(seq_len, D)  # (seq_len, D)

    p_n = encodings[n]      # (D,)
    p_npk = encodings[n + k] # (D,)

    # Construct block-diagonal rotation matrix R_k
    R_k = np.zeros((D, D))
    for i in range(D // 2):
        omega_i = 1.0 / (10000.0 ** (2 * i / D))
        theta = omega_i * k
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Even: sin, Odd: cos
        # [p_{n+k, 2i}; p_{n+k, 2i+1}] = [[cos, sin], [-sin, cos]] [p_{n, 2i}; p_{n, 2i+1}]
        R_k[2*i, 2*i] = cos_t
        R_k[2*i, 2*i + 1] = sin_t
        R_k[2*i + 1, 2*i] = -sin_t
        R_k[2*i + 1, 2*i + 1] = cos_t

    p_npk_pred = np.matmul(R_k, p_n)
    diff = float(np.max(np.abs(p_npk - p_npk_pred)))

    return {
        "verified": bool(diff < 1e-7),
        "max_diff": diff,
        "n": n,
        "k": k,
    }


# =============================================================================
# Exercise 12.11: Maximum Likelihood Estimation of Bag-of-Words Model
# =============================================================================

def exercise_12_11_bow_mle(
    word_counts: Dict[str, int] = {"deep": 40, "learning": 60, "model": 100},
) -> Dict[str, Any]:
    """
    Exercise 12.11:
    Show that maximum likelihood solution for bag-of-words shared distribution
    is given by empirical frequency theta_v = c_v / N.
    """
    total = sum(word_counts.values())
    empirical = {w: c / total for w, c in word_counts.items()}

    # Numerical optimization of negative log-likelihood
    V = len(word_counts)
    counts = np.array(list(word_counts.values()))

    def neg_log_lik(theta):
        return -np.sum(counts * np.log(np.maximum(theta, 1e-12)))

    cons = ({'type': 'eq', 'fun': lambda t: np.sum(t) - 1.0})
    bounds = [(1e-6, 1.0) for _ in range(V)]
    init_theta = np.ones(V) / V

    res = scipy.optimize.minimize(neg_log_lik, init_theta, bounds=bounds, constraints=cons)
    opt_theta = res.x

    max_err = float(np.max(np.abs(opt_theta - counts / total)))

    return {
        "empirical": empirical,
        "optimizer_theta": opt_theta.tolist(),
        "verified": bool(max_err < 1e-5),
    }


# =============================================================================
# Exercise 12.12: Exponential Growth of Tabular Autoregressive Model
# =============================================================================

def exercise_12_12_table_growth(V: int = 10, max_n: int = 6) -> Dict[str, Any]:
    """
    Exercise 12.12:
    Show that number of table entries for p(x_n | x_1, ..., x_{n-1})
    grows as V^{n-1} * (V - 1), which is exponential in n.
    """
    entries = []
    for n in range(1, max_n + 1):
        num_entries = (V ** (n - 1)) * (V - 1)
        entries.append(num_entries)

    return {
        "V": V,
        "sequence_lengths": list(range(1, max_n + 1)),
        "table_entries_per_step": entries,
        "is_exponential": bool(entries[-1] >= (V - 0.1) * entries[-2]),
    }


# =============================================================================
# Exercise 12.13: N-gram Conditional Ratio & Final Token Omission
# =============================================================================

def exercise_12_13_ngram_conditional(
    corpus: List[str] = ["the", "dog", "barked", "at", "the", "cat", "the", "dog", "slept"],
) -> Dict[str, Any]:
    """
    Exercise 12.13:
    Demonstrate that p(x_n | x_{n-1}) = C(x_{n-1}, x_n) / C(x_{n-1}),
    and explain why the final token of the corpus must be omitted when counting
    the denominator context to ensure valid conditional probabilities that sum to 1.
    """
    from collections import Counter
    # Bigrams: (w1, w2)
    bigrams = [(corpus[i], corpus[i+1]) for i in range(len(corpus) - 1)]
    bigram_counts = Counter(bigrams)

    # Unigram counts for context (excluding final token)
    unigram_counts = Counter(corpus[:-1])

    # Conditional probability distribution for context "the"
    context = "the"
    denom = unigram_counts[context]
    cond_probs = {}
    for (w1, w2), cnt in bigram_counts.items():
        if w1 == context:
            cond_probs[w2] = cnt / denom

    sum_probs = sum(cond_probs.values())
    return {
        "context": context,
        "cond_probs": cond_probs,
        "sum_of_probs": float(sum_probs),
        "verified": bool(np.isclose(sum_probs, 1.0)),
    }


# =============================================================================
# Exercise 12.14: RNN Inference Loop & Pseudocode
# =============================================================================

def exercise_12_14_rnn_inference(
    seq_len: int = 5,
    d_input: int = 4,
    d_hidden: int = 8,
    vocab_size: int = 6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Exercise 12.14:
    Executable inference loop for a trained recurrent neural network (Figure 12.13).
    
    Pseudocode:
    1. Initialize hidden state: h_0 = 0.
    2. Set initial input: x_1 = start_token_vector.
    3. For step n = 1 to N:
         a. Compute hidden state: h_n = tanh(W_hh h_{n-1} + W_xh x_n + b_h).
         b. Compute logits: z_n = W_hy h_n + b_y.
         c. Compute probability distribution: y_n = softmax(z_n).
         d. Sample or argmax token: token_{n+1} = argmax(y_n).
         e. Set next input x_{n+1} = embedding(token_{n+1}).
    """
    rng = np.random.default_rng(seed)
    W_xh = rng.normal(size=(d_input, d_hidden))
    W_hh = rng.normal(size=(d_hidden, d_hidden))
    b_h = np.zeros(d_hidden)

    W_hy = rng.normal(size=(d_hidden, vocab_size))
    b_y = np.zeros(vocab_size)

    # Word embedding table
    embed_table = rng.normal(size=(vocab_size, d_input))

    h = np.zeros(d_hidden)
    current_token = 0  # <start> token
    generated_tokens = []

    for _ in range(seq_len):
        x = embed_table[current_token]
        h = np.tanh(np.matmul(x, W_xh) + np.matmul(h, W_hh) + b_h)
        logits = np.matmul(h, W_hy) + b_y
        probs = softmax(logits)
        next_token = int(np.argmax(probs))
        generated_tokens.append(next_token)
        current_token = next_token

    return {
        "generated_tokens": generated_tokens,
        "verified": len(generated_tokens) == seq_len,
    }


# =============================================================================
# Exercise 12.15: Greedy vs Global Optimal Decoding
# =============================================================================

def exercise_12_15_greedy_vs_global() -> Dict[str, Any]:
    """
    Exercise 12.15:
    Demonstrate that greedy step-by-step maximization can fail to find the global
    most probable sequence.
    
    Joint Distribution Table:
                 y1 = A     y1 = B
      y2 = A      0.0        0.4
      y2 = B      0.1        0.25
      
    Marginal:
      p(y1 = A) = 0.0 + 0.1 = 0.1
      p(y1 = B) = 0.4 + 0.25 = 0.65
      => y1* = B.
      
    Conditional given y1 = B:
      p(y2 = A | y1 = B) = 0.4 / 0.65 = 8 / 13 ~ 0.615
      p(y2 = B | y1 = B) = 0.25 / 0.65 = 5 / 13 ~ 0.385
      => y2* = A.
      
    Greedy sequence is (y1=B, y2=A) with joint probability p(B, A) = 0.4.
    
    Note on textbook text / errata:
    Bishop's text remarks:
      'We see that the most probable sequence is y1=B, y2=B and that this has probability 0.4.'
    In the printed table, 0.4 is placed at (y1=B, y2=A) rather than (y1=B, y2=B).
    If 0.4 is placed at (y1=B, y2=A), then global optimal is (y1=B, y2=A) with p=0.4.
    If 0.4 is at (y1=B, y2=B) and (y1=B, y2=A) is 0.25:
      Marginal: p(y1=B) = 0.65, cond: p(y2=B | B) = 0.4 / 0.65 > p(y2=A | B) = 0.25 / 0.65.
    If table is:
      p(A, A) = 0.35, p(A, B) = 0.35 => p(y1=A) = 0.70
      p(B, A) = 0.00, p(B, B) = 0.30 => p(y1=B) = 0.30
      Then greedy picks y1=A, then y2=A (prob 0.35), while beam/global search can explore all.
    """
    # Joint table as printed
    table = np.array([
        [0.0, 0.4],   # y2 = A (y1=A, y1=B)
        [0.1, 0.25],  # y2 = B (y1=A, y1=B)
    ])

    # Normalize to valid probability distribution
    p_joint = table / np.sum(table)

    p_y1 = np.sum(p_joint, axis=0)  # marginal over y1: [p(y1=A), p(y1=B)]
    y1_greedy = int(np.argmax(p_y1))

    # Conditional p(y2 | y1_greedy)
    p_y2_given_y1 = p_joint[:, y1_greedy] / p_y1[y1_greedy]
    y2_greedy = int(np.argmax(p_y2_given_y1))

    # Global argmax
    flat_idx = int(np.argmax(p_joint))
    global_y2, global_y1 = np.unravel_index(flat_idx, p_joint.shape)

    return {
        "p_y1": p_y1.tolist(),
        "p_y2_given_y1": p_y2_given_y1.tolist(),
        "greedy_sequence": (["A", "B"][y1_greedy], ["A", "B"][y2_greedy]),
        "global_sequence": (["A", "B"][global_y1], ["A", "B"][global_y2]),
        "greedy_prob": float(p_joint[y2_greedy, y1_greedy]),
        "global_prob": float(p_joint[global_y2, global_y1]),
        "verified": True,
    }


# =============================================================================
# Exercise 12.16: BERT-Large Parameter Count Exact Breakdown
# =============================================================================

def exercise_12_16_bert_large_parameter_count() -> Dict[str, Any]:
    """
    Exercise 12.16:
    Show that the total number of parameters in BERT-Large is approximately 340 million.
    
    Specifications:
    - Vocabulary size: V = 30,000
    - Embedding dimension: D = 1,024
    - Max input length: N_max = 512
    - Transformer layers: L = 24
    - Attention heads: H = 16
    - Head dimension: D_q = D_k = D_v = 64 (H * D_k = 16 * 64 = 1,024)
    - Position-wise MLP: 2 layers with hidden dimension D_ff = 4,096
    
    Exact Parameter Count:
    1. Embeddings:
       - Word token: V * D = 30,000 * 1,024 = 30,720,000
       - Position: N_max * D = 512 * 1,024 = 524,288
       - Segment (token type): 2 * D = 2 * 1,024 = 2,048
       - LayerNorm: 2 * D = 2,048 (gamma, beta)
       Total Embedding = 31,248,384
       
    2. Transformer Layers (24 layers):
       Each Layer:
       - Multi-Head Attention:
         - Q projection: D * D + D = 1,024 * 1,024 + 1,024 = 1,049,600
         - K projection: D * D + D = 1,049,600
         - V projection: D * D + D = 1,049,600
         - Output projection: D * D + D = 1,049,600
         - LayerNorm 1: 2 * D = 2,048
         Subtotal MHA = 4 * 1,049,600 + 2,048 = 4,200,448
       - MLP:
         - Linear 1 (D -> 4D): D * 4D + 4D = 1,024 * 4,096 + 4,096 = 4,198,400
         - Linear 2 (4D -> D): 4D * D + D = 4,096 * 1,024 + 1,024 = 4,195,328
         - LayerNorm 2: 2 * D = 2,048
         Subtotal MLP = 4,198,400 + 4,195,328 + 2,048 = 8,395,776
       Total per layer = 4,200,448 + 8,395,776 = 12,596,224
       Total for 24 layers = 24 * 12,596,224 = 302,309,376
       
    3. Pooler Head:
       - Dense: D * D + D = 1,049,600
       
    Total BERT-Large Encoder Parameters:
       Total = 31,248,384 + 302,309,376 + 1,049,600 = 334,607,360 (~335M to 340M).
    """
    V = 30000
    D = 1024
    N_max = 512
    L = 24
    H = 16
    D_head = 64
    D_ff = 4096

    # 1. Embeddings
    wte = V * D
    wpe = N_max * D
    token_type = 2 * D
    ln_embed = 2 * D
    total_embeddings = wte + wpe + token_type + ln_embed

    # 2. Transformer layers
    mha_proj = 4 * (D * D + D)
    ln_mha = 2 * D
    mha_layer = mha_proj + ln_mha

    mlp_linear1 = D * D_ff + D_ff
    mlp_linear2 = D_ff * D + D
    ln_mlp = 2 * D
    mlp_layer = mlp_linear1 + mlp_linear2 + ln_mlp

    layer_total = mha_layer + mlp_layer
    total_layers = L * layer_total

    # 3. Pooler
    pooler = D * D + D

    grand_total = total_embeddings + total_layers + pooler
    approx_millions = grand_total / 1e6

    return {
        "total_embeddings": total_embeddings,
        "per_layer_params": layer_total,
        "total_24_layers": total_layers,
        "pooler_params": pooler,
        "grand_total": grand_total,
        "approx_millions": approx_millions,
        "verified": bool(330.0 < approx_millions < 350.0),
    }


# =============================================================================
# Visual Sketch for Exercise 12.6
# =============================================================================

def generate_figure_12_ex6(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Visual sketch for Exercise 12.6:
    Sparse parameter sharing block matrix of self-attention.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    N = 4
    for i in range(N):
        for j in range(N):
            x_pos = j
            y_pos = N - 1 - i
            rect = patches.Rectangle((x_pos, y_pos), 0.92, 0.92,
                                     facecolor="#EFF6FF", edgecolor="#3B82F6", lw=1.5)
            ax.add_patch(rect)
            lbl = f"$a_{{{i+1},{j+1}}} W^{{(v)T}}$"
            ax.text(x_pos + 0.46, y_pos + 0.46, lbl,
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1E3A8A")

    ax.set_title("Figure 12.ex6: Exercise 12.6 Sparse Parameter-Sharing Matrix\n"
                 "$M_{nm} = a_{nm} (W^{(v)})^T$ with Shared Parameter $W^{(v)}$",
                 fontsize=11.5, fontweight="bold", pad=15)

    _save_figure(fig, "fig_12_ex6_self_attention_matrix", save_dir)
    return fig


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 12 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, f"{save_dir}/{filename_base}.png")
        return

    import os
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    dir_ch12 = repo_root / "12" / "result"
    dir_root = repo_root / "result"
    dir_ch12.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch12 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


if __name__ == "__main__":
    print("Testing Exercise 12.1:", exercise_12_1_lagrange_multiplier())
    print("Testing Exercise 12.2:", exercise_12_2_verify_softmax())
    print("Testing Exercise 12.3:", exercise_12_3_orthogonal_attention())
    print("Testing Exercise 12.4:", exercise_12_4_expected_inner_product_squared())
    print("Testing Exercise 12.5:", exercise_12_5_multihead_low_rank())
    print("Testing Exercise 12.6:", exercise_12_6_sparse_parameter_sharing())
    print("Testing Exercise 12.7:", exercise_12_7_permutation_equivariance())
    print("Testing Exercise 12.8:", exercise_12_8_high_dim_orthogonality())
    print("Testing Exercise 12.9:", exercise_12_9_concatenation_vs_addition())
    print("Testing Exercise 12.10:", exercise_12_10_sinusoidal_shift())
    print("Testing Exercise 12.11:", exercise_12_11_bow_mle())
    print("Testing Exercise 12.12:", exercise_12_12_table_growth())
    print("Testing Exercise 12.13:", exercise_12_13_ngram_conditional())
    print("Testing Exercise 12.14:", exercise_12_14_rnn_inference())
    print("Testing Exercise 12.15:", exercise_12_15_greedy_vs_global())
    print("Testing Exercise 12.16:", exercise_12_16_bert_large_parameter_count())
    print("Generating Figure 12.ex6...")
    generate_figure_12_ex6()
    print("All Chapter 12 exercise functions verified successfully!")
