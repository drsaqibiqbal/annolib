"""Intensity normalization for prostate MRI — Robust Anatomy-Referenced
Normalization (RARN).

WHY THIS EXISTS
---------------
MRI intensities are not quantitative: unlike CT Hounsfield units, the same
tissue takes different raw values across scanners, coils, and sequences.
Before any cross-study analysis, intensities must be mapped to a common
frame. The standard whole-image z-score does this poorly for pelvic MRI: the
statistics are contaminated by air, background, and a variable field of view.

METHOD (see prostate_deid_NORMALIZATION_MATH document for the full write-up)
---------------------------------------------------------------------------
RARN standardizes the image using robust location/scale estimated from a
REFERENCE REGION rather than the whole volume:

    reference set R = { I(x) : x in mask }         (prostate gland, or a zone)
                   or the Otsu foreground if no mask is supplied.

    location  mu    = median(R)                    (breakdown point 50%)
    scale     sigma = 1.4826 * MAD(R)              (consistent with SD at Gauss)
              MAD(R) = median(|I(x) - median(R)|)

    normalized image:  I~(x) = ( clip(I(x), q_lo, q_hi) - mu ) / sigma

with q_lo, q_hi the winsorization percentiles of R. The affine map
(mu, sigma, q_lo, q_hi) is logged to the audit trail, so the transform is
AUDIT-REVERSIBLE: on the unclipped support, I(x) = I~(x)*sigma + mu.

The novel elements — relative to established reference-region normalization
(e.g. WhiteStripe for brain MRI) and robust statistics — are (a) referencing
the prostate gland / zonal labels this library already carries, (b) the
optional zone-aware mode that standardizes peripheral and transition zones
separately, and (c) recording the exact affine in the same audit trail as the
de-identification steps, making the whole preprocessing chain reproducible
and invertible. The robust estimators themselves (median, MAD, the 1.4826
constant) are standard and are NOT claimed as novel.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np

from .core import AuditEntry, Sample, Transform
from .organ_mask import WHOLE_GLAND

_MAD_TO_SIGMA = 1.4826  # median(|x-med|) * 1.4826 is a consistent SD estimator
                        # for Gaussian data (1/Phi^{-1}(0.75)).


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method: the intensity threshold maximizing between-class
    variance, used to separate MRI foreground from air/background when no
    anatomical mask is available. numpy-only."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return lo
    p = hist / total
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2))
    return float(centers[idx])


def _robust_stats(ref: np.ndarray) -> Tuple[float, float]:
    """Return (median, 1.4826 * MAD) with a guard against zero scale."""
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    sigma = _MAD_TO_SIGMA * mad
    if sigma <= 0:  # degenerate (near-constant) reference; fall back to SD
        sigma = float(np.std(ref)) or 1.0
    return med, sigma


class IntensityNormalize(Transform):
    """Robust Anatomy-Referenced Normalization for prostate MRI.

    method:
        "robust"     -> median / (1.4826*MAD) standardization  (default)
        "zscore"     -> mean / SD on the winsorized reference
        "minmax"     -> map [q_lo, q_hi] -> [0, 1]
        "zone_aware" -> standardize PZ, TZ, lesion separately, aligning each
                        zone to N(0,1); voxels outside zones use global robust
                        stats. Removes inter-scan zone-level intensity drift,
                        at the cost of altering inter-zone contrast — use when
                        the downstream task is per-zone.
    reference_labels: mask labels defining the reference region (default the
        whole gland). Ignored if the Sample has no mask.
    clip_percentiles: (low, high) winsorization percentiles of the reference.
    use_otsu_foreground: when no mask is present, estimate the reference from
        the Otsu foreground instead of the whole volume.
    """

    def __init__(
        self,
        method: str = "robust",
        reference_labels: Iterable[int] = WHOLE_GLAND,
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        use_otsu_foreground: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if method not in {"robust", "zscore", "minmax", "zone_aware"}:
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self.reference_labels = set(reference_labels)
        self.clip_percentiles = clip_percentiles
        self.use_otsu_foreground = use_otsu_foreground

    # -- reference selection -------------------------------------------------
    def _reference(self, sample: Sample) -> Tuple[np.ndarray, str]:
        img = sample.image
        if sample.mask is not None:
            keep = np.isin(sample.mask, list(self.reference_labels))
            if keep.any():
                ref = img[:, keep] if img.ndim == sample.mask.ndim + 1 else img[keep]
                return ref.ravel(), "mask"
        flat = img.ravel()
        if self.use_otsu_foreground:
            thr = _otsu_threshold(flat)
            fg = flat[flat >= thr]
            if fg.size > 0:
                return fg, "otsu_foreground"
        return flat, "whole_image"

    def apply(self, sample: Sample) -> Sample:
        if self.method == "zone_aware":
            return self._apply_zone_aware(sample)

        ref, ref_kind = self._reference(sample)
        q_lo, q_hi = np.percentile(ref, list(self.clip_percentiles))
        clipped_ref = np.clip(ref, q_lo, q_hi)

        if self.method == "minmax":
            span = (q_hi - q_lo) or 1.0
            sample.image = (np.clip(sample.image, q_lo, q_hi) - q_lo) / span
            mu, sigma = q_lo, span
        else:
            if self.method == "robust":
                mu, sigma = _robust_stats(ref)
            else:  # zscore on winsorized reference
                mu = float(np.mean(clipped_ref))
                sigma = float(np.std(clipped_ref)) or 1.0
            sample.image = (np.clip(sample.image, q_lo, q_hi) - mu) / sigma

        sample.image_dirty = True
        sample.log(
            AuditEntry(
                step="intensity_normalize",
                action=f"normalized_{self.method}",
                detail=(
                    f"reference={ref_kind}, mu={mu:.4g}, sigma={sigma:.4g}, "
                    f"clip=({q_lo:.4g},{q_hi:.4g}), reversible_affine=I*sigma+mu"
                ),
            )
        )
        return sample

    def _apply_zone_aware(self, sample: Sample) -> Sample:
        if sample.mask is None:
            # No zones to reference -> fall back to global robust normalization.
            fallback = IntensityNormalize(method="robust", always_apply=True)
            return fallback.apply(sample)

        out = np.array(sample.image, dtype=np.float32, copy=True)
        g_ref, _ = self._reference(sample)
        g_mu, g_sigma = _robust_stats(g_ref)
        # Default everything to the global standardization first.
        out = (out - g_mu) / g_sigma

        details = [f"global(mu={g_mu:.4g},sigma={g_sigma:.4g})"]
        for label in sorted(self.reference_labels):
            zone = sample.mask == label
            if not zone.any():
                continue
            zvals = sample.image[:, zone] if sample.image.ndim == sample.mask.ndim + 1 else sample.image[zone]
            mu, sigma = _robust_stats(zvals.ravel())
            zsel = (
                np.broadcast_to(zone, out.shape)
                if out.ndim == sample.mask.ndim + 1
                else zone
            )
            src = sample.image
            out[zsel] = ((src[zsel] - mu) / sigma).astype(np.float32)
            details.append(f"label{label}(mu={mu:.4g},sigma={sigma:.4g})")

        sample.image = out
        sample.image_dirty = True
        sample.log(
            AuditEntry(
                step="intensity_normalize",
                action="normalized_zone_aware",
                detail="; ".join(details),
            )
        )
        return sample
