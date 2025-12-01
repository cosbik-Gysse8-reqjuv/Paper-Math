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

# ---------- g(x) = x^2 and derivative ----------

def g(x):
    """Delta-method transform: g(x) = x^2."""
    return x**2

def gprime(x):
    """Derivative g'(x) = 2x."""
    return 2.0 * x

# ---------- delta-method CI coverage + Kolmogorov distance ----------

def delta_ci_coverage_and_D(dist, n, reps, true_mu, true_sigma, rng):
    """
    For a given distribution and sample size n:
      - simulate reps samples of size n
      - compute theta_hat = g(Xbar) = (Xbar)^2
      - build 95% delta-method CIs:
            theta_hat ± 1.96 * |g'(Xbar)| * S / sqrt(n)
        where S is the sample SD
      - report coverage for theta_0 = g(true_mu) = 1
      - standardize using tau = |g'(true_mu)| * true_sigma = 2 * true_sigma:
            Z_n = sqrt(n) * (theta_hat - theta_0) / tau
      - compute Kolmogorov distance of Z_n from N(0,1).
    """
    # simulate data: reps x n
    if dist == 'chisq_scaled':
        # X = (1/3) * chi^2_3
        X = rng.chisquare(df=3, size=(reps, n)) / 3.0
    elif dist == 'uniform':
        # U ~ Uniform(0,2)
        X = rng.uniform(0.0, 2.0, size=(reps, n))
    else:
        raise ValueError("Unknown distribution label")

    # sample means and sample SDs
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)

    # true theta and asymptotic sd tau
    theta0 = g(true_mu)            # = 1
    gprime_mu = gprime(true_mu)    # = 2
    tau = abs(gprime_mu) * true_sigma   # = 2 * sigma

    # delta-method point estimate and plug-in SE
    theta_hat = g(xbar)
    se_hat = np.abs(gprime(xbar)) * s / np.sqrt(n)   # 2*|xbar| * s / sqrt(n)

    # 95% delta-method CIs
    lower = theta_hat - 1.96 * se_hat
    upper = theta_hat + 1.96 * se_hat

    covered = (lower <= theta0) & (theta0 <= upper)
    coverage_prob = covered.mean()

    # standardized delta-method statistic
    Z_n = np.sqrt(n) * (theta_hat - theta0) / tau
    D_hat = kolmogorov_distance(Z_n, norm.cdf)

    return coverage_prob, D_hat

# ---------- run the simulation study ----------

rng = np.random.default_rng(2025)

reps = 100000
n_values = [10, 20, 40, 80, 160, 320]

# true mean and sigmas of ORIGINAL X
mu_true = 1.0
sigma_chi  = np.sqrt(2/3)   # Var = 2/3 for chi^2(3)/3
sigma_unif = np.sqrt(1/3)   # Var = 1/3 for U(0,2)

# Berry–Esseen constants C * B
C_BE = 0.4748
B_unif = 3 * np.sqrt(3) / 4          # = rho / sigma^3 for U(0,2)
B_chi  = 2.1638035505                # from numerical integration for chi^2(3)/3

K_unif = C_BE * B_unif               # ≈ 0.617
K_chi  = C_BE * B_chi                # ≈ 1.027

print("Delta-method estimator with g(x) = x^2")
print("theta_0 = g(1) = 1\n")
print("n   |  Chi^2(3)/3:  cov,  D_hat,  BE(n)   ||  U(0,2):  cov,  D_hat,  BE(n)")
print("----+-------------------------------------------------++--------------------------------")

for n in n_values:
    cov_chi, D_chi = delta_ci_coverage_and_D('chisq_scaled', n, reps,
                                             mu_true, sigma_chi, rng)
    cov_unif, D_unif = delta_ci_coverage_and_D('uniform', n, reps,
                                               mu_true, sigma_unif, rng)

    BE_chi_n  = K_chi  / np.sqrt(n)
    BE_unif_n = K_unif / np.sqrt(n)

    print(f"{n:3d} |  {100*cov_chi:5.1f}%  {D_chi:6.3f}  {BE_chi_n:6.3f}  ||"
          f"  {100*cov_unif:5.1f}%  {D_unif:6.3f}  {BE_unif_n:6.3f}")
