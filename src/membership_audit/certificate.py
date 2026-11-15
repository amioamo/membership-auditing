"""Finite-sample bounds for a cross-population membership audit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import beta, binom, hypergeom, norm


def clopper_pearson_upper(b: int | np.ndarray, m: int, delta: float) -> float | np.ndarray:
    """One-sided Clopper–Pearson upper bound at level 1-delta."""
    b_arr = np.asarray(b, dtype=float)
    scalar = b_arr.ndim == 0
    b_arr = np.atleast_1d(b_arr)
    out = np.empty_like(b_arr, dtype=float)
    full = b_arr >= m
    empty = b_arr <= 0
    mid = ~(full | empty)
    out[full] = 1.0
    if np.any(empty):
        # P(Bin(m,p)=0) = (1-p)^m <= delta => p <= 1 - delta^(1/m)
        out[empty] = 1.0 - delta ** (1.0 / m)
    if np.any(mid):
        out[mid] = beta.ppf(1.0 - delta, b_arr[mid] + 1.0, m - b_arr[mid])
    return float(out[0]) if scalar else out


def robust_order_index(m0: int, v: float, delta0: float) -> int:
    """Largest k in {1..m0} with P(Bin(m0,v) < k) <= delta0; else 0."""
    if m0 <= 0 or not (0.0 < v < 1.0):
        return 0
    # P(N < k) = cdf(k-1)
    # find max k with cdf(k-1) <= delta0
    ks = np.arange(1, m0 + 1)
    cdf = binom.cdf(ks - 1, m0, v)
    ok = ks[cdf <= delta0]
    if ok.size == 0:
        return 0
    return int(ok[-1])


def median_aggregation_delta(n_redraws: int = 21, final_delta: float = 0.05) -> float:
    """Per-draw failure δ such that Pr{Bin(R,δ) ≥ ceil(R/2)} = final_delta.

    For odd R=2m-1 the median fails iff at least m draws fail.
    """
    if n_redraws < 1 or not (0.0 < final_delta < 1.0):
        raise ValueError("n_redraws>=1 and final_delta in (0,1) required")
    m = n_redraws // 2 + 1  # majority threshold
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        # P(X >= m) = sf(m-1)
        p = float(binom.sf(m - 1, n_redraws, mid))
        if p > final_delta:
            hi = mid
        else:
            lo = mid
    return float(lo)


# Default for R=21 → final 95% median coverage (Pr{Bin(21,δ)≥11}=0.05).
MEDIAN_DELTA_R21 = median_aggregation_delta(21, 0.05)


def hypergeom_order_index(m0: int, N0: int, beta: float, delta0: float) -> int:
    """Largest k with P(Hypergeom(N0, K_beta, m0) < k) <= delta0.

    K_beta = ceil(N0 * (1-beta)), matching the finite-reference LCB.
    """
    if m0 <= 0 or N0 <= 0 or not (0.0 < beta < 1.0):
        return 0
    K = int(np.ceil(N0 * (1.0 - beta)))
    K = int(np.clip(K, 0, N0))
    ks = np.arange(1, m0 + 1)
    # P(H < k) = cdf(k-1)
    cdf = hypergeom.cdf(ks - 1, N0, K, m0)
    ok = ks[cdf <= delta0]
    if ok.size == 0:
        return 0
    return int(ok[-1])


def empirical_quantile(scores: np.ndarray, level: float) -> float:
    """Left-continuous empirical quantile via order statistics (numpy definition)."""
    if scores.size == 0:
        return float("nan")
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(scores, level, method="inverted_cdf"))


def audit_tpr(member_scores: np.ndarray, threshold: float) -> float:
    if member_scores.size == 0:
        return float("nan")
    return float(np.mean(member_scores > threshold))


def population_t_alpha(f0_ppf, f1_sf, alpha: float) -> float:
    """T_alpha(F0,F1) = P(S1 > F0^{-1}(1-alpha))."""
    q = float(f0_ppf(1.0 - alpha))
    return float(f1_sf(q))


def population_t_star_kol(f0_ppf, f1_sf, alpha: float, rho: float) -> float:
    """Kolmogorov outer envelope endpoint: P(S1 > F0^{-1}(1-alpha-rho))."""
    v = 1.0 - alpha - rho
    if v <= 0.0:
        return 1.0
    if v >= 1.0:
        return 0.0
    q = float(f0_ppf(v))
    return float(f1_sf(q))


@dataclass
class CertificateResult:
    name: str
    t_hat: float
    u: float
    threshold_used: float
    nontrivial: bool
    extra: Optional[dict] = None


def nominal_estimate(ref: np.ndarray, mem: np.ndarray, alpha: float) -> CertificateResult:
    thr = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr)
    return CertificateResult("naive", t_hat, t_hat, thr, nontrivial=True)


def member_only_ci(ref: np.ndarray, mem: np.ndarray, alpha: float, delta: float) -> CertificateResult:
    thr = empirical_quantile(ref, 1.0 - alpha)
    hits = int(np.sum(mem > thr))
    t_hat = hits / len(mem)
    u = float(clopper_pearson_upper(hits, len(mem), delta))
    return CertificateResult("member_only_ci", t_hat, u, thr, nontrivial=u < 1.0)


def robust_order_stat_certificate(
    ref: np.ndarray,
    mem: np.ndarray,
    alpha: float,
    rho: float,
    delta0: float,
    delta1: float,
    name: str = "robust_os",
) -> CertificateResult:
    m0 = len(ref)
    m1 = len(mem)
    v = 1.0 - alpha - rho
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr_nom)
    if v <= 0.0 or v >= 1.0:
        return CertificateResult(name, t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    k = robust_order_index(m0, v, delta0)
    if k <= 0:
        return CertificateResult(name, t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    ordered = np.sort(ref)
    L = float(ordered[k - 1])  # 1-indexed order statistic
    hits = int(np.sum(mem > L))
    u = float(clopper_pearson_upper(hits, m1, delta1))
    return CertificateResult(
        name,
        t_hat,
        u,
        L,
        nontrivial=u < 1.0 - 1e-15,
        extra={"k": k, "v": v, "hits": hits, "m0": m0, "m1": m1},
    )


def dkw_hoeffding_certificate(
    ref: np.ndarray,
    mem: np.ndarray,
    alpha: float,
    rho: float,
    delta0: float,
    delta1: float,
) -> CertificateResult:
    """DKW band on F0 plus binomial UCB on members (loose global baseline)."""
    m0 = len(ref)
    m1 = len(mem)
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr_nom)
    eps = float(np.sqrt(np.log(2.0 / delta0) / (2.0 * m0)))
    # Lower CI for F0^{-1}(1-alpha-rho): use empirical quantile at 1-alpha-rho-eps
    level = 1.0 - alpha - rho - eps
    if level <= 0.0:
        return CertificateResult("dkw_hoeffding", t_hat, 1.0, float("-inf"), nontrivial=False, extra={"eps": eps})
    L = empirical_quantile(ref, level)
    hits = int(np.sum(mem > L))
    # Member-side: report Clopper–Pearson under the same δ1 as Robust OS.
    # Hoeffding is retained only as a diagnostic in extra[].
    u_hoff = min(1.0, (hits / m1) + np.sqrt(np.log(1.0 / delta1) / (2.0 * m1)))
    u_cp = float(clopper_pearson_upper(hits, m1, delta1))
    u = float(u_cp)
    return CertificateResult(
        "dkw_hoeffding",
        t_hat,
        u,
        float(L),
        nontrivial=u < 1.0,
        extra={"eps": eps, "level": level, "u_hoff": u_hoff, "u_cp": u_cp},
    )


def bootstrap_upper(
    ref: np.ndarray,
    mem: np.ndarray,
    alpha: float,
    rho: float,
    delta: float,
    n_boot: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> CertificateResult:
    """
    Nested bootstrap upper bound for the Kolmogorov envelope endpoint.
    Resamples reference and members; for each draw uses empirical T at level alpha+rho
    as a plug-in for the envelope (standard practice; may undercover at low FPR).
    """
    rng = rng or np.random.default_rng(0)
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr_nom)
    m0, m1 = len(ref), len(mem)
    stats = np.empty(n_boot, dtype=float)
    level = 1.0 - alpha - rho
    for b in range(n_boot):
        r = ref[rng.integers(0, m0, size=m0)]
        m = mem[rng.integers(0, m1, size=m1)]
        if level <= 0:
            stats[b] = 1.0
        else:
            thr = empirical_quantile(r, level)
            stats[b] = audit_tpr(m, thr)
    u = float(np.quantile(stats, 1.0 - delta, method="higher"))
    return CertificateResult(
        "bootstrap",
        t_hat,
        u,
        thr_nom,
        nontrivial=u < 1.0,
        extra={"n_boot": n_boot},
    )


def all_methods(
    ref: np.ndarray,
    mem: np.ndarray,
    alpha: float,
    rho: float,
    delta0: float,
    delta1: float,
    include_bootstrap: bool = True,
    n_boot: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, CertificateResult]:
    delta = delta0 + delta1
    out = {
        "naive": nominal_estimate(ref, mem, alpha),
        "member_only_ci": member_only_ci(ref, mem, alpha, delta),
        "nominal_os": robust_order_stat_certificate(ref, mem, alpha, 0.0, delta0, delta1, name="nominal_os"),
        "dkw_hoeffding": dkw_hoeffding_certificate(ref, mem, alpha, rho, delta0, delta1),
        "robust_os": robust_order_stat_certificate(ref, mem, alpha, rho, delta0, delta1, name="robust_os"),
    }
    if include_bootstrap:
        out["bootstrap"] = bootstrap_upper(ref, mem, alpha, rho, delta, n_boot=n_boot, rng=rng)
    return out


def mc_binomial_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson-ish via Clopper–Pearson two-sided for reporting MC coverage CI."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if successes <= 0 else float(beta.ppf(alpha / 2.0, successes, n - successes + 1))
    hi = 1.0 if successes >= n else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    return lo, hi


def hypergeom_kup(b: int, n: int, N: int, delta: float) -> int:
    """Largest K with Pr(Hypergeom(N,K,n) <= b) > delta (exact inversion UCB)."""
    if N <= 0 or n <= 0:
        return 0
    b = int(np.clip(b, 0, n))
    best = 0
    for K in range(0, N + 1):
        # Support may be empty for extreme K; treat as degenerate.
        lo = max(0, n - (N - K))
        hi = min(n, K)
        if b < lo:
            p = 0.0
        elif b >= hi:
            p = 1.0
        else:
            p = float(hypergeom.cdf(b, N, K, n))
        if p > delta:
            best = K
    return int(best)


def hypergeom_upper(b: int, n: int, N: int, delta: float) -> float:
    """One-sided finite-population upper bound K_up/N."""
    return float(hypergeom_kup(b, n, N, delta)) / float(N)


def robust_order_stat_census(
    ref: np.ndarray,
    mem: np.ndarray,
    alpha: float,
    rho: float,
    delta0: float,
) -> CertificateResult:
    """Census member-side bound: U = (# members > L) / N with δ1=0."""
    m0 = len(ref)
    N = len(mem)
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr_nom)
    v = 1.0 - alpha - rho
    if v <= 0.0 or v >= 1.0:
        return CertificateResult("robust_os_census", t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    k = robust_order_index(m0, v, delta0)
    if k <= 0:
        return CertificateResult("robust_os_census", t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    L = float(np.sort(ref)[k - 1])
    hits = int(np.sum(mem > L))
    u = hits / N
    return CertificateResult(
        "robust_os_census",
        t_hat,
        float(u),
        L,
        nontrivial=u < 1.0,
        extra={"k": k, "hits": hits, "N": N},
    )


def robust_order_stat_hypergeom(
    ref: np.ndarray,
    mem_sample: np.ndarray,
    N: int,
    alpha: float,
    rho: float,
    delta0: float,
    delta1: float,
) -> CertificateResult:
    """Threshold LCB + hypergeometric member-side UCB for sample size n from N."""
    m0 = len(ref)
    n = len(mem_sample)
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    # plug-in on the sample only (diagnostic)
    t_hat = audit_tpr(mem_sample, thr_nom)
    v = 1.0 - alpha - rho
    if v <= 0.0 or v >= 1.0:
        return CertificateResult("robust_os_hg", t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    k = robust_order_index(m0, v, delta0)
    if k <= 0:
        return CertificateResult("robust_os_hg", t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    L = float(np.sort(ref)[k - 1])
    hits = int(np.sum(mem_sample > L))
    u = hypergeom_upper(hits, n, N, delta1)
    return CertificateResult(
        "robust_os_hg",
        t_hat,
        float(u),
        L,
        nontrivial=u < 1.0,
        extra={"k": k, "hits": hits, "n": n, "N": N},
    )


def finite_ref_order_stat_census(
    ref: np.ndarray,
    mem: np.ndarray,
    N0: int,
    alpha: float,
    rho: float,
    delta0: float,
    name: str = "finite_ref_census",
) -> CertificateResult:
    """Without-replacement finite-reference LCB + member census U.

    ``ref`` is an SRS of size m0 from a fixed pool of size N0; the LCB uses
    the hypergeometric order-statistic construction.
    """
    m0 = len(ref)
    N = len(mem)
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr_nom)
    beta = alpha + rho
    if beta <= 0.0 or beta >= 1.0:
        return CertificateResult(name, t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    k = hypergeom_order_index(m0, N0, beta, delta0)
    if k <= 0:
        return CertificateResult(name, t_hat, 1.0, float("-inf"), nontrivial=False, extra={"k": 0})
    L = float(np.sort(ref)[k - 1])
    hits = int(np.sum(mem > L))
    u = hits / N
    return CertificateResult(
        name,
        t_hat,
        float(u),
        L,
        nontrivial=u < 1.0,
        extra={
            "k": k,
            "hits": hits,
            "N": N,
            "N0": N0,
            "m0": m0,
            "n_distinct": int(len(np.unique(ref))),
        },
    )


def median_aggregate(values: list[float] | np.ndarray) -> float:
    return float(np.median(np.asarray(values, dtype=float)))
