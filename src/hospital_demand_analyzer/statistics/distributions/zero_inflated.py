"""Zero-inflated Poisson (ZIP) and Zero-Inflated Negative Binomial (ZINB)."""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

ZIP_NAME = "Zero-Inflated Poisson"
ZINB_NAME = "Zero-Inflated Negative Binomial"


def _as_array(k):
    return np.asarray(k, dtype=float)


class ZeroInflatedPoisson:
    """P(X=0)=pi+(1-pi)*Pois(0); P(X=k>0)=(1-pi)*Pois(k), k>0."""

    def pmf(self, k, pi: float, lam: float):
        k = _as_array(k)
        out = np.zeros_like(k, dtype=float)
        mask0 = k == 0
        mask_pos = k > 0
        out[mask0] = pi + (1.0 - pi) * stats.poisson.pmf(0, lam)
        if np.any(mask_pos):
            out[mask_pos] = (1.0 - pi) * stats.poisson.pmf(k[mask_pos], lam)
        return out

    def logpmf(self, k, pi: float, lam: float):
        k = _as_array(k)
        out = np.empty_like(k, dtype=float)
        for i, ki in enumerate(np.atleast_1d(k).flat):
            if ki == 0:
                val = pi + (1.0 - pi) * math.exp(-lam)
                out.flat[i] = math.log(max(val, 1e-300))
            else:
                out.flat[i] = math.log(max(1.0 - pi, 1e-300)) + stats.poisson.logpmf(int(ki), lam)
        return out if np.ndim(k) else float(out.flat[0])

    def cdf(self, k, pi: float, lam: float):
        k = _as_array(k)
        out = np.zeros_like(k, dtype=float)
        for i, ki in enumerate(np.atleast_1d(k).flat):
            kk = int(math.floor(float(ki)))
            out.flat[i] = float(np.sum(self.pmf(np.arange(0, kk + 1), pi, lam)))
        return out if np.ndim(k) else float(out.flat[0])

    def mean(self, pi: float, lam: float) -> float:
        return (1.0 - pi) * lam

    def rvs(self, pi: float, lam: float, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        size = 1 if size is None else size
        structural = rng.random(size) < pi
        counts = stats.poisson.rvs(lam, size=size, random_state=rng)
        out = np.where(structural, 0, counts)
        if size == 1:
            return int(np.asarray(out).item())
        return out


class ZeroInflatedNBinom:
    """P(X=0)=pi+(1-pi)*NB(0); P(X=k>0)=(1-pi)*NB(k), k>0 (scipy n,p)."""

    def pmf(self, k, pi: float, n: float, p: float):
        k = _as_array(k)
        out = np.zeros_like(k, dtype=float)
        mask0 = k == 0
        mask_pos = k > 0
        out[mask0] = pi + (1.0 - pi) * stats.nbinom.pmf(0, n, p)
        if np.any(mask_pos):
            out[mask_pos] = (1.0 - pi) * stats.nbinom.pmf(k[mask_pos], n, p)
        return out

    def logpmf(self, k, pi: float, n: float, p: float):
        k = _as_array(k)
        out = np.empty_like(k, dtype=float)
        for i, ki in enumerate(np.atleast_1d(k).flat):
            if ki == 0:
                val = pi + (1.0 - pi) * stats.nbinom.pmf(0, n, p)
                out.flat[i] = math.log(max(val, 1e-300))
            else:
                out.flat[i] = math.log(max(1.0 - pi, 1e-300)) + stats.nbinom.logpmf(int(ki), n, p)
        return out if np.ndim(k) else float(out.flat[0])

    def cdf(self, k, pi: float, n: float, p: float):
        k = _as_array(k)
        out = np.zeros_like(k, dtype=float)
        for i, ki in enumerate(np.atleast_1d(k).flat):
            kk = int(math.floor(float(ki)))
            out.flat[i] = float(np.sum(self.pmf(np.arange(0, kk + 1), pi, n, p)))
        return out if np.ndim(k) else float(out.flat[0])

    def mean(self, pi: float, n: float, p: float) -> float:
        return (1.0 - pi) * stats.nbinom.mean(n, p)

    def rvs(self, pi: float, n: float, p: float, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        size = 1 if size is None else size
        structural = rng.random(size) < pi
        counts = stats.nbinom.rvs(n, p, size=size, random_state=rng)
        out = np.where(structural, 0, counts)
        if size == 1:
            return int(np.asarray(out).item())
        return out


zip_dist = ZeroInflatedPoisson()
zinb_dist = ZeroInflatedNBinom()


def _zip_loglik(data: np.ndarray, pi: float, lam: float) -> float:
    return float(np.sum(zip_dist.logpmf(data, pi, lam)))


def _zinb_loglik(data: np.ndarray, pi: float, n: float, p: float) -> float:
    return float(np.sum(zinb_dist.logpmf(data, pi, n, p)))


def fit_zip(data: np.ndarray) -> tuple[dict[str, float], float, bool]:
    int_data = np.round(data).astype(int)
    int_data = int_data[int_data >= 0]
    if len(int_data) < 5:
        return {}, float("-inf"), False

    lam0 = max(float(np.mean(int_data[int_data > 0])) if np.any(int_data > 0) else 0.5, 0.05)
    pi0 = min(max(float(np.mean(int_data == 0)) * 0.5, 0.01), 0.99)

    def neg_ll(theta):
        pi = 1.0 / (1.0 + math.exp(-theta[0]))
        lam = math.exp(theta[1])
        if pi <= 0 or pi >= 1 or lam <= 0:
            return 1e12
        return -_zip_loglik(int_data, pi, lam)

    x0 = [math.log(pi0 / (1.0 - pi0)), math.log(lam0)]
    try:
        res = optimize.minimize(neg_ll, x0, method="L-BFGS-B")
        if not res.success:
            return {}, float("-inf"), False
        pi = 1.0 / (1.0 + math.exp(-res.x[0]))
        lam = math.exp(res.x[1])
        log_lik = _zip_loglik(int_data, pi, lam)
        return {"pi": pi, "lambda": lam}, log_lik, True
    except Exception:
        return {}, float("-inf"), False


def fit_zinb(data: np.ndarray) -> tuple[dict[str, float], float, bool]:
    int_data = np.round(data).astype(int)
    int_data = int_data[int_data >= 0]
    if len(int_data) < 5:
        return {}, float("-inf"), False

    pos = int_data[int_data > 0]
    if len(pos) < 3:
        return {}, float("-inf"), False

    mean_pos = float(np.mean(pos))
    var_pos = float(np.var(pos))
    if var_pos <= mean_pos or mean_pos <= 0:
        return {}, float("-inf"), False

    p0 = mean_pos / var_pos
    n0 = mean_pos * p0 / (1.0 - p0)
    if n0 <= 0 or p0 <= 0 or p0 >= 1:
        return {}, float("-inf"), False

    pi0 = min(max(float(np.mean(int_data == 0)) * 0.5, 0.01), 0.99)

    def neg_ll(theta):
        pi = 1.0 / (1.0 + math.exp(-theta[0]))
        n = math.exp(theta[1])
        p = 1.0 / (1.0 + math.exp(-theta[2]))
        if pi <= 0 or pi >= 1 or n <= 0 or p <= 0 or p >= 1:
            return 1e12
        return -_zinb_loglik(int_data, pi, n, p)

    x0 = [
        math.log(pi0 / (1.0 - pi0)),
        math.log(max(n0, 0.1)),
        math.log(p0 / (1.0 - p0)),
    ]
    try:
        res = optimize.minimize(neg_ll, x0, method="L-BFGS-B")
        if not res.success:
            return {}, float("-inf"), False
        pi = 1.0 / (1.0 + math.exp(-res.x[0]))
        n = math.exp(res.x[1])
        p = 1.0 / (1.0 + math.exp(-res.x[2]))
        log_lik = _zinb_loglik(int_data, pi, n, p)
        return {"pi": pi, "n": n, "p": p}, log_lik, True
    except Exception:
        return {}, float("-inf"), False
