import numpy as np


# ============================================================
# basic utils
# ============================================================

def _normalize_rows(x, eps=1e-15):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def _stable_log_sinh(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 20.0
    out[small] = np.log(np.sinh(x[small]))
    xs = x[~small]
    out[~small] = xs - np.log(2.0) + np.log1p(-np.exp(-2.0 * xs))
    return out


def _as_batch_vec(x, B, name):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1 or x.shape[0] != B:
        raise ValueError(f"{name} must have shape [B] or [B,1], got {x.shape}")
    return x


# ============================================================
# 1) exact batched vMF coverage on S^2
# ============================================================

def vmf_coverage_s2_batch(target_x, mu, kappa):
    """
    Exact HPD coverage for batched vMF fits on S^2.

    Parameters
    ----------
    target_x : [B,3]
    mu       : [B,3]
    kappa    : [B] or [B,1]

    Returns
    -------
    coverage : [B]
        c_b = P_{Y~vMF(mu_b,kappa_b)}[ p(Y) >= p(target_x_b) ]
    """
    target_x = np.asarray(target_x, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if target_x.ndim != 2 or target_x.shape[1] != 3:
        raise ValueError(f"target_x must have shape [B,3], got {target_x.shape}")
    if mu.shape != target_x.shape:
        raise ValueError(f"mu must have shape [B,3], got {mu.shape}")

    B = target_x.shape[0]
    kappa = _as_batch_vec(kappa, B, "kappa")

    target_x = _normalize_rows(target_x)
    mu = _normalize_rows(mu)

    z = np.sum(target_x * mu, axis=1)
    z = np.clip(z, -1.0, 1.0)

    out = np.empty(B, dtype=float)

    mask0 = (kappa == 0.0)
    out[mask0] = 1.0

    mask = ~mask0
    if np.any(mask):
        kk = kappa[mask]
        zz = z[mask]
        # c = (e^k - e^{kz}) / (e^k - e^{-k})
        # stable form:
        # c = (1 - exp(k(z-1))) / (1 - exp(-2k))
        num = 1.0 - np.exp(kk * (zz - 1.0))
        den = 1.0 - np.exp(-2.0 * kk)
        out[mask] = num / den

    return np.clip(out, 0.0, 1.0)


# ============================================================
# 2) batched ZLP-Kent helpers
# canonical model:
#   Fisher zoom + diag(u,1/u,1) + rotation with cols (g2,g3,g1)
# ============================================================

def _batch_rotation_from_gammas(gamma1, gamma2, gamma3):
    """
    Build batched rotation matrices R with columns (gamma2, gamma3, gamma1).
    Shape: [B,3,3]
    """
    gamma1 = _normalize_rows(gamma1)

    gamma2 = gamma2 - np.sum(gamma2 * gamma1, axis=1, keepdims=True) * gamma1
    gamma2 = _normalize_rows(gamma2)

    g3 = np.cross(gamma1, gamma2)
    g3 = _normalize_rows(g3)

    flip = np.sum(g3 * gamma3, axis=1) < 0.0
    gamma2 = gamma2.copy()
    g3 = g3.copy()
    gamma2[flip] *= -1.0
    g3[flip] *= -1.0

    return np.stack([gamma2, g3, gamma1], axis=-1)  # [B,3,3]


def zlpkent_logpdf_s2_batch(target_x, gamma1, gamma2, gamma3, kappa, u):
    """
    Exact batched logpdf for the fitted ZLP-Kent model on S^2.

    Parameters
    ----------
    target_x : [B,3]
    gamma1   : [B,3]
    gamma2   : [B,3]
    gamma3   : [B,3]
    kappa    : [B] or [B,1]
    u        : [B] or [B,1]

    Returns
    -------
    logpdf   : [B]
    """
    target_x = np.asarray(target_x, dtype=float)
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    gamma3 = np.asarray(gamma3, dtype=float)

    if target_x.ndim != 2 or target_x.shape[1] != 3:
        raise ValueError(f"target_x must have shape [B,3], got {target_x.shape}")

    B = target_x.shape[0]
    if gamma1.shape != (B, 3) or gamma2.shape != (B, 3) or gamma3.shape != (B, 3):
        raise ValueError("gamma1, gamma2, gamma3 must all have shape [B,3]")

    kappa = _as_batch_vec(kappa, B, "kappa")
    u = _as_batch_vec(u, B, "u")

    target_x = _normalize_rows(target_x)
    R = _batch_rotation_from_gammas(gamma1, gamma2, gamma3)

    # target -> canonical: y = x @ R
    Y = np.einsum("bi,bij->bj", target_x, R)
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    y3 = Y[:, 2]

    inv_u = 1.0 / u
    r2 = (y1 * inv_u) ** 2 + (y2 * u) ** 2 + y3 * y3
    r = np.sqrt(np.clip(r2, 1e-300, None))
    z_base = y3 / r

    log_norm = np.log(kappa) - np.log(4.0 * np.pi) - _stable_log_sinh(kappa)
    return log_norm + kappa * z_base - 1.5 * np.log(r2)


def sample_zlpkent_s2_batch(gamma1, gamma2, gamma3, kappa, u, n_ref, seed=0):
    """
    Batched stable sampler for ZLP-Kent.

    Returns
    -------
    samples : [B,n_ref,3]
    """
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma2 = np.asarray(gamma2, dtype=float)
    gamma3 = np.asarray(gamma3, dtype=float)

    B = gamma1.shape[0]
    if gamma1.shape != (B, 3) or gamma2.shape != (B, 3) or gamma3.shape != (B, 3):
        raise ValueError("gamma vectors must all have shape [B,3]")

    kappa = _as_batch_vec(kappa, B, "kappa")
    u = _as_batch_vec(u, B, "u")

    rng = np.random.default_rng(seed)

    # uniform base on S^2
    base = rng.normal(size=(B, n_ref, 3))
    base /= np.linalg.norm(base, axis=2, keepdims=True)

    x0 = base[:, :, 0]
    y0 = base[:, :, 1]
    z0 = np.clip(base[:, :, 2], -1.0, 1.0)

    kk = kappa[:, None]

    # stable Fisher zoom
    log_term = np.logaddexp(np.log1p(z0), np.log1p(-z0) - 2.0 * kk)
    z1 = 1.0 + (log_term - np.log(2.0)) / kk
    z1 = np.clip(z1, -1.0, 1.0)

    phi = np.arctan2(y0, x0)
    rho1 = np.sqrt(np.clip(1.0 - z1 * z1, 0.0, None))

    zoom = np.empty_like(base)
    zoom[:, :, 0] = rho1 * np.cos(phi)
    zoom[:, :, 1] = rho1 * np.sin(phi)
    zoom[:, :, 2] = z1

    # LP/project with A = diag(u,1/u,1)
    uu = u[:, None]
    y1 = uu * zoom[:, :, 0]
    y2 = (1.0 / uu) * zoom[:, :, 1]
    y3 = zoom[:, :, 2]

    norm = np.sqrt(y1 * y1 + y2 * y2 + y3 * y3)

    can = np.empty_like(base)
    can[:, :, 0] = y1 / norm
    can[:, :, 1] = y2 / norm
    can[:, :, 2] = y3 / norm

    # canonical -> target
    R = _batch_rotation_from_gammas(gamma1, gamma2, gamma3)
    target = np.einsum("bnj,bij->bni", can, R)
    return target


# ============================================================
# 3) batched Monte Carlo HPD coverage for ZLP-Kent
# ============================================================

def coverage_from_logpdf_samples(ref_logpdf, target_logpdf, weights=None):
    """
    Generic empirical HPD-coverage estimator from reference log-pdf samples.

    Parameters
    ----------
    ref_logpdf : array, shape (..., M)
        Log-pdf values of reference samples drawn from the distribution of interest.
        The last axis is the sample axis.

    target_logpdf : array, shape (...) or (..., K)
        Log-pdf values of target points whose contained coverage you want.
        Leading batch dimensions must match ref_logpdf.shape[:-1].

    weights : array, shape (..., M), optional
        Optional nonnegative weights for the reference samples.
        Use this if your reference samples are not equally weighted.
        If None, all samples get equal weight.

    Returns
    -------
    coverage : array, shape matching target_logpdf
        Empirical coverage values in [0,1], defined as the estimated mass of
        points with log-pdf >= target_logpdf.

    Notes
    -----
    If ref_logpdf comes from actual samples Y_j ~ p, and weights=None, then this estimates

        c(x) = P_{Y~p}[ log p(Y) >= log p(x) ].

    This is exactly the HPD-style coverage quantity used for P-P curves.

    If you have target_logpdf shape (...,), the output has shape (...,).
    If you have target_logpdf shape (..., K), the output has shape (..., K).
    """

    ref_logpdf = np.asarray(ref_logpdf, dtype=float)
    target_logpdf = np.asarray(target_logpdf, dtype=float)

    if ref_logpdf.ndim < 1:
        raise ValueError("ref_logpdf must have at least 1 dimension")

    batch_shape = ref_logpdf.shape[:-1]
    M = ref_logpdf.shape[-1]

    # target can be batch-shaped (...) or batch+query-shaped (..., K)
    if target_logpdf.shape[:len(batch_shape)] != batch_shape:
        raise ValueError(
            f"Leading dimensions of target_logpdf must match ref_logpdf.shape[:-1]. "
            f"Got ref batch shape {batch_shape}, target shape {target_logpdf.shape}."
        )

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != ref_logpdf.shape:
            raise ValueError(
                f"weights must have same shape as ref_logpdf. "
                f"Got weights {weights.shape}, ref_logpdf {ref_logpdf.shape}."
            )
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")

    # Flatten batch dims so implementation is simple and robust
    B = int(np.prod(batch_shape)) if batch_shape else 1
    ref_flat = ref_logpdf.reshape(B, M)

    target_tail_shape = target_logpdf.shape[len(batch_shape):]
    if len(target_tail_shape) == 0:
        K = 1
        target_flat = target_logpdf.reshape(B, 1)
        squeeze_last = True
    else:
        K = int(np.prod(target_tail_shape))
        target_flat = target_logpdf.reshape(B, K)
        squeeze_last = False

    if weights is None:
        sorted_ref = np.sort(ref_flat, axis=1)  # ascending
        out = np.empty((B, K), dtype=float)

        for b in range(B):
            idx = np.searchsorted(sorted_ref[b], target_flat[b], side="left")
            out[b] = 1.0 - idx / M

    else:
        w_flat = weights.reshape(B, M)
        out = np.empty((B, K), dtype=float)

        for b in range(B):
            order = np.argsort(ref_flat[b])
            sref = ref_flat[b, order]
            sw = w_flat[b, order]

            wsum = sw.sum()
            if wsum <= 0:
                raise ValueError(f"weights sum to zero in batch item {b}")

            sw = sw / wsum
            cdf_lt = np.concatenate([[0.0], np.cumsum(sw)])  # mass of entries < insertion point

            idx = np.searchsorted(sref, target_flat[b], side="left")
            out[b] = 1.0 - cdf_lt[idx]

    out = np.clip(out, 0.0, 1.0)

    if batch_shape:
        if squeeze_last:
            return out.reshape(batch_shape)
        return out.reshape(batch_shape + target_tail_shape)
    else:
        if squeeze_last:
            return out.reshape(())
        return out.reshape(target_tail_shape)


def zlp_kent_coverage(target_samples, gamma1, gamma2, gamma3, kappa, u, num_samples_per_bitem=10000, seed=0):

    B = gamma1.shape[0]
    if gamma1.shape != (B, 3) or gamma2.shape != (B, 3) or gamma3.shape != (B, 3):
        raise ValueError("gamma vectors must all have shape [B,3]")

    ref_samples=sample_zlpkent_s2_batch(gamma1, gamma2, gamma3, kappa, u, num_samples_per_bitem, seed=seed)
    M=num_samples_per_bitem

    ref_logpdf = zlpkent_logpdf_s2_batch(
        ref_samples.reshape(B * M, 3),
        np.repeat(gamma1, M, axis=0),
        np.repeat(gamma2, M, axis=0),
        np.repeat(gamma3, M, axis=0),
        np.repeat(kappa, M, axis=0),
        np.repeat(u, M, axis=0),
    ).reshape(B, M)

    target_logpdf = zlpkent_logpdf_s2_batch(
        target_samples, gamma1, gamma2, gamma3, kappa, u
    )

    coverage = coverage_from_logpdf_samples(ref_logpdf, target_logpdf)

    return coverage
