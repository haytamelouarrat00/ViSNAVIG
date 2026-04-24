"""Photometric building blocks for the Direct Visual Servoing controller.

Each helper is a pure function so it can be unit-tested in isolation. The DVS
controller composes them; nothing here knows about cameras or scenes.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


# -----------------------------------------------------------------------------
# Image gradients
# -----------------------------------------------------------------------------

# 7-tap Farid derivative kernel (URAI2015 eq. 8): more accurate than Sobel/Scharr
# for photometric VS, especially at sub-pixel residuals.
_FARID7_DERIV = np.array(
    [-112.0, -913.0, -2047.0, 0.0, 2047.0, 913.0, 112.0]
) / 8418.0
# Matched 7-tap smoothing kernel (Farid & Simoncelli, IEEE TIP 2004 Table II).
_FARID7_SMOOTH = np.array(
    [0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711]
)


def farid7_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """7-tap Farid derivative filter — returns (Ix, Iy) in pixel-intensity units."""
    I = image.astype(np.float64)
    Ix = cv2.sepFilter2D(I, cv2.CV_64F, _FARID7_DERIV, _FARID7_SMOOTH)
    Iy = cv2.sepFilter2D(I, cv2.CV_64F, _FARID7_SMOOTH, _FARID7_DERIV)
    return Ix, Iy


def scharr_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scharr derivative filter — kept as an alternative."""
    I = image.astype(np.float64)
    Ix = cv2.Scharr(I, cv2.CV_64F, 1, 0) / 32.0
    Iy = cv2.Scharr(I, cv2.CV_64F, 0, 1) / 32.0
    return Ix, Iy


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Cast to float64 grayscale."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return image.astype(np.float64)


# -----------------------------------------------------------------------------
# Pyramids
# -----------------------------------------------------------------------------

def image_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Gaussian pyramid, finest first: [full, /2, /4, ...]."""
    pyr = [image.astype(np.float64) if image.dtype != np.float64 else image]
    for _ in range(max(0, levels - 1)):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def depth_pyramid(depth: np.ndarray, levels: int) -> list[np.ndarray]:
    """Depth pyramid by area-averaging (avoids pyrDown's Gaussian smearing of holes)."""
    pyr = [depth.astype(np.float64)]
    for _ in range(max(0, levels - 1)):
        d = pyr[-1]
        h, w = d.shape
        # cv2.resize INTER_AREA averages, but preserves NaNs only if we mask.
        valid = np.isfinite(d) & (d > 0)
        d_safe = np.where(valid, d, 0.0)
        w_mask = valid.astype(np.float64)
        d_down = cv2.resize(d_safe, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        w_down = cv2.resize(w_mask, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        out = np.where(w_down > 1e-6, d_down / np.maximum(w_down, 1e-6), np.nan)
        pyr.append(out)
    return pyr


def intrinsics_pyramid(K: np.ndarray, levels: int) -> list[np.ndarray]:
    """K rescaled to match cv2.pyrDown's pixel-center convention.

    cv2.pyrDown maps (u_new, v_new) → (2*u_new, 2*v_new) on the source grid, i.e.
    fx_new = fx/2, cx_new = (cx + 0.5)/2 - 0.5 (and same for fy/cy).
    """
    pyrs = [K.astype(np.float64).copy()]
    K_l = pyrs[0]
    for _ in range(max(0, levels - 1)):
        K_next = K_l.copy()
        K_next[0, 0] *= 0.5
        K_next[1, 1] *= 0.5
        K_next[0, 2] = (K_l[0, 2] + 0.5) * 0.5 - 0.5
        K_next[1, 2] = (K_l[1, 2] + 0.5) * 0.5 - 0.5
        pyrs.append(K_next)
        K_l = K_next
    return pyrs


# -----------------------------------------------------------------------------
# Pixel validity
# -----------------------------------------------------------------------------

def hole_mask(
    depth: np.ndarray,
    border: int = 10,
    dilate: int = 4,
    grad_quantile: float = 0.99,
) -> np.ndarray:
    """Boolean mask of pixels safe to use in L_I.

    URAI2015 §3 motivates the neighbour-validity check: pixels at depth holes
    or strong depth discontinuities give meaningless gradients in the rendered
    image. We:
      1. drop NaN / non-positive depths,
      2. dilate the invalid set so a pixel near a hole is also dropped,
      3. drop pixels whose depth gradient magnitude is in the top quantile
         (depth discontinuities → occlusion edges),
      4. crop a uniform border so derivative kernels never touch the edge.
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-3)

    if dilate > 0:
        invalid = (~valid).astype(np.uint8)
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        invalid = cv2.dilate(invalid, kernel)
        valid = invalid == 0

    if 0.0 < grad_quantile < 1.0 and valid.any():
        d_safe = np.where(valid, depth, float(np.median(depth[valid])))
        gx = cv2.Sobel(d_safe, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(d_safe, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        thresh = float(np.quantile(grad_mag[valid], grad_quantile))
        valid = valid & (grad_mag <= thresh)

    if border > 0:
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False

    return valid


# -----------------------------------------------------------------------------
# Luminance interaction matrix (Collewet & Marchand RR-6631 eq. 9)
# -----------------------------------------------------------------------------

def luminance_interaction_matrix(
    Ix: np.ndarray,
    Iy: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel L_I with shape (N_pix, 6) and a (N_pix,) boolean validity mask.

    L_I rows for invalid pixels are set to zero (so they contribute nothing to
    the normal equations) but kept in place so callers can index back into the
    image grid by reshape((H, W, 6)).
    """
    h, w = Ix.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = np.isfinite(depth) & (depth > 1e-3)
    if mask is not None:
        valid = valid & mask

    Z = np.where(valid, depth, 1.0)
    inv_z = 1.0 / Z

    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64), indexing="xy")
    x = (u - cx) / fx
    y = (v - cy) / fy

    # Standard 2×6 point interaction matrix L_x rows (one per pixel)
    Lx_rows = np.stack(
        [-inv_z, np.zeros_like(x), x * inv_z, x * y, -(1.0 + x * x), y],
        axis=-1,
    )
    Ly_rows = np.stack(
        [np.zeros_like(y), -inv_z, y * inv_z, 1.0 + y * y, -x * y, -x],
        axis=-1,
    )

    # Image gradients live in pixel units; L_x is in normalised coords. Scale.
    Ix_n = (fx * Ix)[..., None]
    Iy_n = (fy * Iy)[..., None]

    # L_I = -∇I^T · L_x (RR-6631 eq. 9)
    L_I = -(Ix_n * Lx_rows + Iy_n * Ly_rows)

    flat = L_I.reshape(-1, 6)
    valid_flat = valid.reshape(-1)
    flat[~valid_flat] = 0.0
    return flat, valid_flat


# -----------------------------------------------------------------------------
# Photometric normalisation & residuals
# -----------------------------------------------------------------------------

def affine_normalize(I_src: np.ndarray, I_ref: np.ndarray) -> np.ndarray:
    """Linearly map I_src to match I_ref's mean and std (zero-mean affine)."""
    mu_s, sigma_s = float(np.mean(I_src)), float(np.std(I_src))
    mu_r, sigma_r = float(np.mean(I_ref)), float(np.std(I_ref))
    alpha = sigma_r / (sigma_s + 1e-8)
    beta = mu_r - alpha * mu_s
    return (alpha * I_src + beta).astype(np.float64)


def znssd_residual(
    I_curr: np.ndarray, I_target: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Zero-mean normalised SSD residual (URAI2015 §4: most convex of the lot).

    Returns (per-pixel residual, scalar cost). Both inputs must already be the
    same shape and float-valued.
    """
    Ic = I_curr - float(np.mean(I_curr))
    sc = float(np.std(I_curr)) + 1e-8
    It = I_target - float(np.mean(I_target))
    st = float(np.std(I_target)) + 1e-8
    e = (Ic / sc) - (It / st)
    return e, 0.5 * float(np.sum(e * e))


# -----------------------------------------------------------------------------
# Robust weighting (Huber)
# -----------------------------------------------------------------------------

def huber_weights(residual: np.ndarray, k: float = 1.345) -> np.ndarray:
    """IRLS weights from a Huber M-estimator with MAD-estimated scale.

    Returns weights of the same shape as `residual`. Pixels with large
    standardised residual get w_i ≪ 1; the bulk stays at 1.
    """
    r = residual.ravel()
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) + 1e-12
    sigma = 1.4826 * mad
    standardised = np.abs(r - med) / sigma
    w = np.ones_like(r)
    big = standardised > k
    w[big] = k / standardised[big]
    return w.reshape(residual.shape)


# -----------------------------------------------------------------------------
# Mutual Information cost & analytic Jacobian (Dame & Marchand 2011)
# -----------------------------------------------------------------------------

def _bspline3(t: np.ndarray) -> np.ndarray:
    """Cubic B-spline kernel β(t)."""
    a = np.abs(t)
    out = np.zeros_like(t)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    out[m1] = 2.0 / 3.0 - a[m1] ** 2 + 0.5 * a[m1] ** 3
    out[m2] = ((2.0 - a[m2]) ** 3) / 6.0
    return out


def _bspline3_deriv(t: np.ndarray) -> np.ndarray:
    """Derivative β'(t) of the cubic B-spline."""
    a = np.abs(t)
    s = np.sign(t)
    out = np.zeros_like(t)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    out[m1] = s[m1] * (-2.0 * a[m1] + 1.5 * a[m1] ** 2)
    out[m2] = -s[m2] * 0.5 * (2.0 - a[m2]) ** 2
    return out


def mi_cost_grad_hess(
    I_curr: np.ndarray,
    I_target: np.ndarray,
    L_I: np.ndarray,
    valid_mask: np.ndarray,
    n_bins: int = 8,
    intensity_range: Tuple[float, float] = (0.0, 255.0),
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Mutual information cost & its analytic 6-vector gradient + 6×6 Hessian.

    Cost is `-MI(I_curr, I_target)` so that the standard `v = -λ H⁻¹ g` law
    *maximises* MI. Implementation follows Dame & Marchand 2011: cubic B-spline
    Parzen joint histogram, analytic per-pixel chain rule through the bin
    contributions back to L_I.

    Args:
        I_curr, I_target: same-shape float images (any range; rescaled to
            [0, n_bins-2] internally to leave a 1-bin safety margin for the
            B-spline support).
        L_I: (N_pix, 6) per-pixel luminance interaction matrix (already
            zero-rowed for invalid pixels).
        valid_mask: (N_pix,) boolean — only these pixels contribute.
        n_bins: number of histogram bins. ViSP uses 8-16; 8 is faster.
        intensity_range: (min, max) used to map intensities into bin space.

    Returns:
        (cost, grad, hess) where cost is -MI, grad has shape (6,), hess has
        shape (6, 6).
    """
    Ic = I_curr.ravel()
    It = I_target.ravel()
    valid = valid_mask.ravel()

    Ic = Ic[valid]
    It = It[valid]
    L = L_I[valid]
    N = Ic.shape[0]
    if N == 0:
        return 0.0, np.zeros(6), np.eye(6) * 1e-6

    lo, hi = float(intensity_range[0]), float(intensity_range[1])
    # Map to [1, n_bins-2]: leaves a 1-bin margin on each side for the cubic
    # B-spline's support of width 4. Prevents out-of-range bin indices.
    inner = max(1.0, n_bins - 3.0)
    s_c = 1.0 + (Ic - lo) / max(hi - lo, 1e-8) * inner
    s_t = 1.0 + (It - lo) / max(hi - lo, 1e-8) * inner
    # The intensity-to-bin scale (used in the chain-rule derivative).
    bin_scale = inner / max(hi - lo, 1e-8)

    i0 = np.floor(s_c).astype(np.int64)
    j0 = np.floor(s_t).astype(np.int64)

    Nb = int(n_bins)
    hist = np.zeros((Nb, Nb), dtype=np.float64)
    djdv = np.zeros((Nb, Nb, 6), dtype=np.float64)

    # 4×4 = 16 bin pairs per pixel (cubic B-spline support width 4).
    for di in (-1, 0, 1, 2):
        i_idx = i0 + di
        bx = _bspline3(s_c - i_idx.astype(np.float64))
        bx_d = _bspline3_deriv(s_c - i_idx.astype(np.float64))
        for dj in (-1, 0, 1, 2):
            j_idx = j0 + dj
            in_range = (
                (i_idx >= 0) & (i_idx < Nb) & (j_idx >= 0) & (j_idx < Nb)
            )
            if not np.any(in_range):
                continue
            ii = i_idx[in_range]
            jj = j_idx[in_range]
            by = _bspline3(s_t[in_range] - jj.astype(np.float64))

            # Joint histogram contribution.
            np.add.at(hist, (ii, jj), bx[in_range] * by)

            # dp(i,j) / dv contribution. Note: dI/dv = L (RR-6631 sign convention),
            # and ds/dI = bin_scale, so chain rule pulls in bx_d * bin_scale * L.
            grad_contrib = (bx_d[in_range] * by * bin_scale)[:, None] * L[in_range]
            np.add.at(djdv, (ii, jj), grad_contrib)

    # Normalise to probabilities.
    total = hist.sum()
    if total < 1e-12:
        return 0.0, np.zeros(6), np.eye(6) * 1e-6
    p_ij = hist / total
    djdv_p = djdv / total  # ∂p(i,j)/∂v

    p_i = p_ij.sum(axis=1)  # marginal over j (target)
    p_j = p_ij.sum(axis=0)  # marginal over i (current)
    # We want ∂MI/∂v = sum_ij ∂p(i,j)/∂v · (1 + log(p(i,j)/p_i(i))).
    # (The p_j term cancels because sum_j ∂p(i,j)/∂v = ∂p_i(i)/∂v and that
    # falls out when subtracting the marginal entropy gradient.)
    nonzero = p_ij > 1e-12
    log_term = np.zeros_like(p_ij)
    p_i_safe = np.where(p_i > 1e-12, p_i, 1.0)[:, None]
    log_term[nonzero] = 1.0 + np.log(p_ij[nonzero] / np.broadcast_to(p_i_safe, p_ij.shape)[nonzero])

    grad_mi = np.einsum("ij,ijk->k", log_term, djdv_p)

    # Gauss-Newton-ish Hessian approximation (Dame & Marchand): sum 1/p(i,j) · g·gᵀ.
    inv_p = np.where(nonzero, 1.0 / np.maximum(p_ij, 1e-12), 0.0)
    djdv_flat = djdv_p.reshape(-1, 6)
    weights = inv_p.reshape(-1)
    hess_mi = (djdv_flat * weights[:, None]).T @ djdv_flat

    # MI itself (for cost reporting).
    mi = float(np.sum(p_ij[nonzero] * np.log(
        p_ij[nonzero] / (p_i_safe.repeat(Nb, axis=1)[nonzero] *
                         np.broadcast_to(p_j[None, :], p_ij.shape)[nonzero])
    )))

    # We minimise -MI, so flip sign of cost & gradient. Hessian of -MI under the
    # GN approximation is the negative of the GN Hessian of MI, but for a
    # *descent* step we use the positive-definite GN form directly (this is the
    # standard trick: the MI GN Hessian is built as Σ (1/p)·ggᵀ which is PSD,
    # and we add LM damping anyway).
    return -mi, -grad_mi, hess_mi
