"""Mixture classes, induced Kolmogorov radius, and threshold lower bounds."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog

from .certificate import (
    audit_tpr,
    clopper_pearson_upper,
    empirical_quantile,
    robust_order_stat_certificate,
)


def remap_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.vectorize(remap.get)(labels)


def subgroup_cdfs(scores: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    K = int(labels.max()) + 1
    F = np.zeros((K, len(grid)))
    for k in range(K):
        sk = scores[labels == k]
        if sk.size == 0:
            F[k] = np.linspace(0.0, 1.0, len(grid))
            continue
        sk_sorted = np.sort(sk)
        F[k] = np.searchsorted(sk_sorted, grid, side="right") / sk.size
    return F


def mixture_cdf(F_sub: np.ndarray, w: np.ndarray) -> np.ndarray:
    return F_sub.T @ w


def quantile_from_cdf(grid: np.ndarray, cdf: np.ndarray, level: float) -> float:
    if level <= 0.0:
        return float(grid[0])
    if level >= 1.0:
        return float(grid[-1])
    idx = int(np.searchsorted(cdf, level, side="left"))
    idx = min(max(idx, 0), len(grid) - 1)
    return float(grid[idx])


def extremize_linear_over_weight_class(
    c: np.ndarray,
    w0: np.ndarray,
    gamma: float,
    sense: str = "max",
) -> float:
    """
    Optimize c·w over {w in simplex : ||w-w0||_1 <= gamma}.
    Uses an LP with slack variables for the L1 ball.
    """
    c = np.asarray(c, dtype=float)
    w0 = np.asarray(w0, dtype=float)
    K = len(w0)
    if K == 0:
        return 0.0
    if gamma <= 0.0:
        return float(c @ w0)

    # Variables: w[0:K], t[0:K] with t_i >= |w_i - w0_i|.
    c_obj = np.zeros(2 * K)
    c_obj[:K] = -c if sense == "max" else c
    bounds = [(0.0, 1.0)] * K + [(0.0, None)] * K
    A_eq = np.zeros((1, 2 * K))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])
    A_ub = np.zeros((1 + 2 * K, 2 * K))
    b_ub = np.zeros(1 + 2 * K)
    A_ub[0, K:] = 1.0
    b_ub[0] = gamma
    for i in range(K):
        A_ub[1 + i, i] = 1.0
        A_ub[1 + i, K + i] = -1.0
        b_ub[1 + i] = w0[i]
        A_ub[1 + K + i, i] = -1.0
        A_ub[1 + K + i, K + i] = -1.0
        b_ub[1 + K + i] = -w0[i]
    res = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError("mixture weight linear program failed")
    return float(c @ res.x[:K])


def recover_extremal_weight(
    c: np.ndarray,
    w0: np.ndarray,
    gamma: float,
    sense: str = "max",
) -> np.ndarray:
    """Return an optimizing weight for c·w over the mixture class."""
    c = np.asarray(c, dtype=float)
    w0 = np.asarray(w0, dtype=float)
    K = len(w0)
    if gamma <= 0.0 or K == 0:
        return w0.copy()
    c_obj = np.zeros(2 * K)
    c_obj[:K] = -c if sense == "max" else c
    bounds = [(0.0, 1.0)] * K + [(0.0, None)] * K
    A_eq = np.zeros((1, 2 * K))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])
    A_ub = np.zeros((1 + 2 * K, 2 * K))
    b_ub = np.zeros(1 + 2 * K)
    A_ub[0, K:] = 1.0
    b_ub[0] = gamma
    for i in range(K):
        A_ub[1 + i, i] = 1.0
        A_ub[1 + i, K + i] = -1.0
        b_ub[1 + i] = w0[i]
        A_ub[1 + K + i, i] = -1.0
        A_ub[1 + K + i, K + i] = -1.0
        b_ub[1 + K + i] = -w0[i]
    res = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError("mixture weight linear program failed")
    w = np.maximum(res.x[:K], 0.0)
    s = float(w.sum())
    return w / s if s > 0 else w0.copy()


def induced_radius_lp(F_sub: np.ndarray, w0: np.ndarray, gamma: float) -> float:
    """Exact rho_F = sup_w ||F_w - F_{w0}||_inf over the L1 mixture class."""
    F0 = mixture_cdf(F_sub, w0)
    rho = 0.0
    for j in range(F_sub.shape[1]):
        c = F_sub[:, j]
        hi = extremize_linear_over_weight_class(c, w0, gamma, sense="max")
        lo = extremize_linear_over_weight_class(c, w0, gamma, sense="min")
        rho = max(rho, abs(hi - F0[j]), abs(F0[j] - lo))
    return float(rho)


def structured_quantile_lp(
    grid: np.ndarray,
    F_sub: np.ndarray,
    w0: np.ndarray,
    gamma: float,
    level: float,
) -> tuple[float, np.ndarray]:
    """
    Exact q = inf_{w in W} F_w^{-1}(level) for mixture CDFs on a fixed grid.
    Equivalent to the leftmost t with max_w F_w(t) >= level.
    """
    if level <= 0.0:
        return float(grid[0]), w0.copy()
    lo, hi = 0, len(grid) - 1
    best = len(grid) - 1
    w_star = w0.copy()
    while lo <= hi:
        mid = (lo + hi) // 2
        c = F_sub[:, mid]
        val = extremize_linear_over_weight_class(c, w0, gamma, sense="max")
        if val + 1e-12 >= level:
            best = mid
            w_star = recover_extremal_weight(c, w0, gamma, sense="max")
            hi = mid - 1
        else:
            lo = mid + 1
    return float(grid[best]), w_star


def feasible_weights(
    w0: np.ndarray,
    gamma: float,
    n_search: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Legacy random/directed candidates; kept for diagnostics only."""
    K = len(w0)
    cands = [w0.copy()]
    for k in range(K):
        for s in (-1.0, 1.0):
            w = w0.copy()
            w[k] += s * gamma
            w = np.clip(w, 0.0, None)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            if np.sum(np.abs(w - w0)) <= gamma + 1e-8:
                cands.append(w)
    for _ in range(n_search):
        noise = rng.dirichlet(np.ones(K))
        direction = noise - w0
        denom = np.sum(np.abs(direction))
        if denom < 1e-12:
            continue
        step = (rng.random() * gamma) / denom
        w = np.clip(w0 + step * direction, 0.0, None)
        w = w / w.sum()
        if np.sum(np.abs(w - w0)) <= gamma + 1e-6:
            cands.append(w)
    return cands


def _evaluation_grid(scores: np.ndarray, max_points: int | None = None) -> np.ndarray:
    """Every distinct score. An explicit ``max_points`` only reproduces a retired cap."""
    grid = np.unique(np.asarray(scores, dtype=float))
    if max_points is None or grid.size <= max_points:
        return grid
    idx = np.linspace(0, grid.size - 1, max_points).astype(int)
    return grid[idx]


def analyze_structured_class(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    labels: np.ndarray,
    gamma: float,
    alpha: float,
    n_search: int = 2500,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Construct F from subgroup mixtures and compute exact LP endpoints:
      rho_F, T_struct*, T_Kol*(rho_F), T_Kol*(gamma).

    The structured target and the induced radius use every distinct
    non-member score. A failed weight program raises.
    """
    rng = rng or np.random.default_rng(0)
    labels = remap_labels(labels)
    K = int(labels.max()) + 1
    w0 = np.array([(labels == k).mean() for k in range(K)], dtype=float)
    grid = _evaluation_grid(nonmember_scores)
    F_sub = subgroup_cdfs(nonmember_scores, labels, grid)
    F0 = mixture_cdf(F_sub, w0)

    rho_f = induced_radius_lp(F_sub, w0, gamma)
    q_struct, w_q_star = structured_quantile_lp(grid, F_sub, w0, gamma, 1.0 - alpha)
    t_direct = audit_tpr(member_scores, q_struct)

    # Also maximize TPR directly over the same LP quantile construction:
    # for mixture classes the lower-quantile program is the relevant audit endpoint.
    t_struct = t_direct
    w_t_star = w_q_star

    def kol_endpoint(rho: float) -> tuple[float, float]:
        level = 1.0 - alpha - rho
        if level <= 0.0:
            return float(grid[0]), 1.0
        q = quantile_from_cdf(grid, F0, level)
        return q, audit_tpr(member_scores, q)

    # Analytic Kolmogorov outer radius for an L1 mixture budget is gamma/2.
    rho_half = 0.5 * gamma
    q_ind, t_kol_ind = kol_endpoint(rho_f)
    q_half, t_kol_half = kol_endpoint(rho_half)
    q_loose, t_kol_loose = kol_endpoint(gamma)  # naive: treat gamma as score-space radius
    q_nom = quantile_from_cdf(grid, F0, 1.0 - alpha)
    t_nom = audit_tpr(member_scores, q_nom)

    # Optional search diagnostic (not used for reported endpoints).
    cands = feasible_weights(w0, gamma, min(n_search, 200), rng)
    t_search = max(
        (audit_tpr(member_scores, quantile_from_cdf(grid, mixture_cdf(F_sub, w), 1.0 - alpha))
         for w in cands),
        default=t_direct,
    )

    return {
        "K": K,
        "w0": w0,
        "gamma": gamma,
        "alpha": alpha,
        "rho_F": rho_f,
        "rho_half": rho_half,
        "rho_loose": gamma,
        "delta_rho": max(rho_half - rho_f, 0.0),
        "q_nom": q_nom,
        "t_nom": t_nom,
        "q_direct": q_struct,
        "t_direct": t_direct,
        "t_struct_oracle": float(max(t_struct, t_search)),
        "q_induced": q_ind,
        "t_kol_induced": t_kol_ind,
        "q_half": q_half,
        "t_kol_half": t_kol_half,
        "q_loose": q_loose,
        "t_kol_loose": t_kol_loose,
        "delta_T": t_kol_ind - t_direct,
        "delta_T_half": t_kol_half - t_direct,
        "delta_T_loose": t_kol_loose - t_direct,
        "w_q_star": w_q_star,
        "w_t_star": w_t_star,
        "grid": grid,
        "F_sub": F_sub,
        "F0": F0,
    }



def _subgroup_dkw_eps(n_k: np.ndarray, delta0: float) -> np.ndarray:
    """Per-group DKW radii with Bonferroni split over K active groups."""
    n_k = np.asarray(n_k, dtype=float)
    K = int(np.sum(n_k > 0))
    if K <= 0:
        return np.zeros_like(n_k)
    return np.sqrt(np.log(2.0 * K / delta0) / (2.0 * np.maximum(n_k, 1.0)))


def _mixture_cdf_upper_at(
    grid: np.ndarray,
    F_sub: np.ndarray,
    w: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """Pointwise upper envelope sum_k w_k min(1, F_k(t) + eps_k)."""
    return np.minimum(F_sub + eps[:, None], 1.0).T @ w


def mixture_quantile_lcb_for_weight(
    grid: np.ndarray,
    F_sub: np.ndarray,
    w: np.ndarray,
    eps: np.ndarray,
    alpha: float,
) -> float:
    """Lower bound on F_w^{-1}(1-alpha) via mixture upper CDF envelope."""
    level = 1.0 - alpha
    upper = _mixture_cdf_upper_at(grid, F_sub, w, eps)
    if float(upper[0]) >= level:
        return float(grid[0])
    if float(upper[-1]) < level:
        return float("-inf")
    idx = int(np.searchsorted(upper, level, side="left"))
    idx = min(max(idx, 0), len(grid) - 1)
    return float(grid[idx])


def mixture_threshold_lcb(
    ref: np.ndarray,
    labels: np.ndarray,
    gamma: float,
    alpha: float,
    delta0: float,
    n_search: int,
    rng: np.random.Generator,
    w0: np.ndarray | None = None,
    max_points: int | None = None,
) -> float:
    """
    Valid lower confidence bound on q_alpha^-(F) for subgroup-mixture class.
    Uses simultaneous subgroup DKW bands and an exact LP over feasible weights:
    L = inf{t : max_w F_w^{up}(t) >= 1-alpha}.

    The default searches every distinct reference score. The upper-envelope
    map is monotone, so binary search evaluates one LP per probe.
    max_points may be set only to reproduce the retired candidate-grid cap;
    that cap can return a threshold above this lower confidence bound.
    """
    del n_search, rng  # retained for call-site compatibility
    labels = remap_labels(labels)
    K = int(labels.max()) + 1
    n_k = np.array([(labels == k).sum() for k in range(K)], dtype=float)
    if np.any(n_k <= 0):
        return float("-inf")
    eps = _subgroup_dkw_eps(n_k, delta0)
    if w0 is None:
        w0 = n_k / n_k.sum()
    else:
        w0 = np.asarray(w0, dtype=float)
        if w0.shape[0] != K:
            w0 = n_k / n_k.sum()
        else:
            w0 = w0 / w0.sum()
    if max_points is None:
        grid = np.unique(np.asarray(ref, dtype=float))
    else:
        grid = _evaluation_grid(ref, max_points=max_points)
    F_sub = subgroup_cdfs(ref, labels, grid)
    F_up = np.minimum(F_sub + eps[:, None], 1.0)
    level = 1.0 - alpha
    # Leftmost t with max_w F_w^{up}(t) >= 1-alpha.
    lo, hi = 0, len(grid) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        val = extremize_linear_over_weight_class(F_up[:, mid], w0, gamma, sense="max")
        if val + 1e-12 >= level:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return float("-inf")
    return float(grid[best])


def pure_mixture_certificate(
    ref: np.ndarray,
    labels: np.ndarray,
    mem: np.ndarray,
    gamma: float,
    alpha: float,
    delta0: float,
    delta1: float,
    rng: np.random.Generator,
    w0: np.ndarray | None = None,
):
    """Finite-sample certificate using only the mixture threshold LCB."""
    L_mix = mixture_threshold_lcb(
        ref, labels, gamma, alpha, delta0, n_search=0, rng=rng, w0=w0
    )
    thr_nom = empirical_quantile(ref, 1.0 - alpha)
    t_hat = audit_tpr(mem, thr_nom)
    if not np.isfinite(L_mix) or L_mix <= float("-inf"):
        return type(
            "R",
            (),
            {
                "t_hat": t_hat,
                "u": 1.0,
                "threshold_used": float("-inf"),
                "nontrivial": False,
                "extra": {"L_mix": L_mix},
            },
        )()
    hits = int(np.sum(mem > L_mix))
    u = float(clopper_pearson_upper(hits, len(mem), delta1))
    return type(
        "R",
        (),
        {
            "t_hat": t_hat,
            "u": u,
            "threshold_used": L_mix,
            "nontrivial": u < 1.0 - 1e-15,
            "extra": {"L_mix": L_mix},
        },
    )()


def structured_mixture_certificate(
    ref: np.ndarray,
    labels: np.ndarray,
    mem: np.ndarray,
    gamma: float,
    alpha: float,
    rho_f: float,
    delta0: float,
    delta1: float,
    rng: np.random.Generator,
    n_search: int = 400,
    w0: np.ndarray | None = None,
):
    """
    Structure-aware certificate: max of mixture-class threshold LCB and
    induced Kolmogorov OS at rho_F (both valid lower bounds on q^-).
    """
    r_ind = robust_order_stat_certificate(
        ref, mem, alpha, rho_f, delta0, delta1, name="induced"
    )
    L_mix = mixture_threshold_lcb(
        ref, labels, gamma, alpha, delta0, n_search, rng, w0=w0
    )
    L_ind = r_ind.threshold_used
    # Valid lower bound: max of mixture-class and induced Kolmogorov LCBs.
    if not np.isfinite(L_mix) or L_mix <= float("-inf"):
        L = L_ind
    else:
        L = max(L_mix, L_ind)
    hits = int(np.sum(mem > L))
    u = float(clopper_pearson_upper(hits, len(mem), delta1))
    return type(
        "R",
        (),
        {
            "t_hat": r_ind.t_hat,
            "u": u,
            "threshold_used": L,
            "nontrivial": u < 1.0 - 1e-15,
            "extra": {"L_mix": L_mix, "L_ind": L_ind},
        },
    )()


