import numpy as np
from scipy.stats import norm

# ---------- helpers for ECDF and Kolmogorov distance ----------

def ecdf(x):
    """Empirical CDF of a 1D array x."""
    x_sorted = np.sort(x)
    n = len(x_sorted)
    y = np.arange(1, n + 1) / n
    return x_sorted, y

def kolmogorov_distance(samples, cdf):
    """
    Sup_x |F_n(x) - cdf(x)| where F_n is ECDF of samples
    and cdf is a callable (e.g., norm.cdf).
    """
    xs, Fn = ecdf(samples)
    F = cdf(xs)
    return np.max(np.abs(Fn - F))

# ---------- CI coverage + Kolmogorov distance in one function ----------

def ci_coverage_and_D(dist, n, reps, true_mean, true_sigma, rng):
    """
    For a given distribution and n:
      - simulate reps samples of size n
      - compute 95% CIs: xbar ± 1.96 * S / sqrt(n)
      - report coverage
      - standardize means with true sigma and compute Kolmogorov distance
          D_hat = sup_x |F_n(x) - Phi(x)| for Z_n

    dist: 'chisq_scaled' or 'uniform'
    true_mean: E[X]
    true_sigma: sqrt(Var(X))
    """
    # Generate data: reps x n
    if dist == 'chisq_scaled':
        # X = (1/3) * chi^2_3
        X = rng.chisquare(df=200, size=(reps, n)) / 200.0
    elif dist == 'uniform':
        # U ~ Uniform(0,2)
        X = rng.uniform(0.0, 2.0, size=(reps, n))
    else:
        raise ValueError("Unknown distribution label")

    # Sample means and sample standard deviations (ddof=1 for unbiased S)
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)

    # ---------- 95% CI coverage ----------
    se_hat = s / np.sqrt(n)
    lower = xbar - 1.96 * se_hat
    upper = xbar + 1.96 * se_hat

    covered = (lower <= true_mean) & (true_mean <= upper)
    coverage_prob = covered.mean()

    # ---------- Kolmogorov distance for standardized sample mean ----------
    # Z_n = sqrt(n) * (xbar - mu) / true_sigma
    Z_n = np.sqrt(n) * (xbar - true_mean) / true_sigma
    D_hat = kolmogorov_distance(Z_n, norm.cdf)

    return coverage_prob, D_hat

# ---------- run the study ----------

rng = np.random.default_rng(2025)

true_mean = 1.0
reps = 100000                    # 1000 intervals / 1000 Z_n's
n_values = [5, 10, 20, 40, 80, 160, 320]

# true sigmas
sigma_chi = np.sqrt(2/200)       # Var = 2/3 for Chi^2(3)/3
sigma_unif = np.sqrt(1/3)      # Var = 1/3 for Unif(0,2)

print("n   |  Chi^2(3)/3:  coverage,  D_hat   ||  Unif(0,2):  coverage,  D_hat")
print("----+----------------------------------++--------------------------------")

for n in n_values:
    cov_chi, D_chi = ci_coverage_and_D('chisq_scaled', n, reps,
                                       true_mean, sigma_chi, rng)
    cov_unif, D_unif = ci_coverage_and_D('uniform', n, reps,
                                         true_mean, sigma_unif, rng)

    print(f"{n:3d} |   {100*cov_chi:6.2f}%   {D_chi:7.4f}   ||"
          f"   {100*cov_unif:6.2f}%   {D_unif:7.4f}")
