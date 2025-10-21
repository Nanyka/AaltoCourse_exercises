"""estimate_em_obs.py
------------------
Estimator for the dynamic portfolio-choice model with CRRA preferences and
**budget-embedded financial literacy costs** (fixed + ad-valorem).

We estimate a **two-type finite mixture** via **EM** where **type priors depend
on observables** (logistic in [1, age0, ln(a0)]). The dynamic inner problem is
solved by **NFXP** (value iteration) in `nfxp_model.py`.

USAGE (CLI)
===========
python estimate_em_obs.py \
  --csv "/path/to/data.csv" \
  --outdir "./outputs_em_obs" \
  --na 24 --gh 5 --em 6 --nm 30

INPUT DATA
==========
CSV must contain columns: id, t, a, y, x, age
- id   : individual identifier
- t    : time index within individual
- a    : assets entering period t
- y    : income (we use its mean y_bar in the model's first pass)
- x    : observed risky share in [0,1] (we discretize to nearest grid point)
- age  : age (or an age index)

OUTPUTS
=======
(outdir)/estimates.csv          : parameters (beta, sigma, costs by type, eta coefficients, avg prior p(type1))
(outdir)/responsibilities.csv   : per-id posterior type probabilities (omega_type1/2) and aligned priors
(outdir)/V_type*.npy            : value functions on the state grid (for type 1 and 2)
(outdir)/CCP_type*.npy          : choice probabilities on the state grid (for type 1 and 2)
(outdir)/policy_type*.csv       : argmax policies (x*, s*) over the state grid (for type 1 and 2)

METHOD IN A NUTSHELL
====================
EM with observable-dependent priors:
  E-step:
    - Solve the DP for each type with current parameters (NFXP → CCPs).
    - Compute each individual's per-type log-likelihood using P(x|state)
      (savings is unobserved ⇒ we marginalize over s).
    - Combine with the logistic prior p(type1 | Z_i = [1, age0, ln a0]) to
      get responsibilities (posterior probabilities) omega_{i1}, omega_{i2}.
  M-step:
    - Update logistic coefficients eta by weighted logistic regression
      (weights = responsibilities).
    - Update per-type cost parameters by weighted MLE (Nelder–Mead; each
      evaluation re-solves that type's DP).
    - Optionally update shared (beta, sigma) by maximizing the mixture
      likelihood (ECM step).

Design choices for speed/robustness:
  - Small action grids for x and s.
  - Nearest-neighbor projection for next-period assets.
  - Gauss–Hermite quadrature for risky return integration.
  - Transform parameters (beta, sigma) to keep them in sensible bounds.
  - Log-sum-exp and max-subtraction tricks for numerical stability.
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

# Import the dynamic model and parameter container
from nfxp_model_v2 import NFXPFinLitConsBudgetAgeType, SpecType


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Read and clean the input panel.

    Keeps only the required columns and sorts by (id, t) to make the
    per-id likelihood construction simple and robust.
    """
    df = pd.read_csv(csv_path)
    needed = {'id', 't', 'age', 'a', 'y', 'x'}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}. Found {df.columns.tolist()}")
    df = df[['id', 't', 'age', 'a', 'y', 'x']].copy().sort_values(['id', 't']).reset_index(drop=True)
    return df


def prep_grids(df: pd.DataFrame, n_a: int = 24):
    """
    Build discretization grids for the DP and actions.

    - Asset grid: linear between 5th and 95th percentiles (robust to outliers).
    - Age grid: unique ages; if too many, thin to <= 25 using quantiles.
    - Action grids: coarse grids for risky share x and savings rate s.
    - Income: use the panel mean y_bar on this first pass.

    Returns
    -------
    a_grid, age_grid, x_grid, s_grid, y_bar
    """
    A_MIN = max(1.0, float(df['a'].quantile(0.05)))
    A_MAX = float(df['a'].quantile(0.95))
    if A_MAX <= A_MIN:
        A_MAX = A_MIN + 1.0
    a_grid = np.linspace(A_MIN, A_MAX, n_a)

    age_grid = np.sort(df['age'].unique())
    if len(age_grid) > 25:
        qs = np.linspace(0, 1, 25)
        age_grid = np.unique(np.round(np.quantile(df['age'], qs), 3))

    x_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_grid = np.array([0.6, 0.8, 0.95])

    y_bar = float(df['y'].mean())
    return a_grid, age_grid, x_grid, s_grid, y_bar


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct per-id features for the observable-dependent type prior.

    We use Z_i = [1, age0, ln(a0)]. We can expand this (e.g., education)
    by modifying this function and ensuring the CSV contains those columns.
    """
    rows = []
    for pid, sub in df.groupby('id'):
        sub = sub.sort_values('t')
        age0 = float(sub['age'].iloc[0])
        a0 = float(sub['a'].iloc[0])
        ln_a0 = np.log(max(a0, 1e-8))
        rows.append([pid, 1.0, age0, ln_a0])
    Z = pd.DataFrame(rows, columns=['id', 'const', 'age0', 'ln_a0']).set_index('id')
    return Z


def sigmoid(z):
    """Numerically-stable logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def per_id_loglike_for_type(df: pd.DataFrame, model: NFXPFinLitConsBudgetAgeType, CCP: np.ndarray):
    """
    Compute per-id log-likelihood under a given type's CCPs.
    Observed 'x' is matched to nearest point on x_grid.
    Savings 's' is unobserved and integrated out via CCP sum over s.

    Returns
    -------
    ells : array of log-likelihoods per id
    order: array of ids in the same order as ells
    """
    ells = []
    order = []
    for pid, sub in df.groupby('id'):
        sub = sub.sort_values('t')
        ccp_x = model.ccp_x_marginal(sub['a'].to_numpy(), sub['age'].to_numpy(), CCP)
        x_idx = np.array([np.argmin(np.abs(model.xgrid - v)) for v in sub['x']])
        p = ccp_x[np.arange(len(x_idx)), x_idx]
        ells.append(np.sum(np.log(np.maximum(p, 1e-12))))
        order.append(pid)
    return np.asarray(ells), np.asarray(order)


def run_em_obs(csv_path: str, outdir: str = "outputs_em_obs",
               n_a: int = 24, gh_order: int = 5,
               em_iters: int = 6, nm_steps: int = 30, seed: int = 42):
    """
    Run EM with observable-dependent type priors.

    Parameters
    ----------
    csv_path : str
        Path to panel CSV (requires columns: id, t, a, y, x, age).
    outdir : str
        Where to write outputs (estimates, responsibilities, DP arrays, policies).
    n_a : int
        Number of asset grid points.
    gh_order : int
        Gauss–Hermite quadrature order for risky return integration.
    em_iters : int
        Number of EM iterations.
    nm_steps : int
        Max Nelder–Mead steps for each parameter block update.
    seed : int
        RNG seed for initialization of logistic prior coefficients.

    Returns
    -------
    dict with final parameters and average prior Pr(type=1).
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    df = load_data(csv_path)
    a_grid, age_grid, x_grid, s_grid, y_bar = prep_grids(df, n_a=n_a)
    model = NFXPFinLitConsBudgetAgeType(a_grid, age_grid, y_bar, x_grid, s_grid, gh_order=gh_order)

    Zdf = build_features(df)
    ids = Zdf.index.to_numpy()
    Z = Zdf.to_numpy()
    id_to_pos = {pid: i for i, pid in enumerate(ids)}

    beta_sigma_raw = np.array([0.0, 0.0])
    theta1 = np.array([ 1.0, -0.01, -1.0, 0.01])
    theta2 = np.array([ 2.0,  0.01, -2.0, 0.02])
    eta    = rng.normal(0.0, 0.1, size=Z.shape[1])

    base_const = dict(R_f=1.02, mu_lnR=np.log(1.06), sigma_lnR=0.20)

    def pack_shared(bs):
        """Map transformed vector bs to (beta, sigma) with sensible bounds."""
        beta  = 0.50 + 0.49 * (1.0 / (1.0 + np.exp(-bs[0])))
        sigma = 0.50 + 4.50 * (1.0 / (1.0 + np.exp(-bs[1])))
        return beta, sigma

    for em in range(em_iters):
        beta, sigma = pack_shared(beta_sigma_raw)
        spec1 = SpecType(beta=beta, sigma=sigma, **base_const,
                         gamma0=theta1[0], gamma1=theta1[1], delta0=theta1[2], delta1=theta1[3])
        spec2 = SpecType(beta=beta, sigma=sigma, **base_const,
                         gamma0=theta2[0], gamma1=theta2[1], delta0=theta2[2], delta1=theta2[3])

        V1, CCP1, _, _ = model.value_iteration_one_type(spec1)
        V2, CCP2, _, _ = model.value_iteration_one_type(spec2)

        ell1_vec, id_order1 = per_id_loglike_for_type(df, model, CCP1)
        ell2_vec, id_order2 = per_id_loglike_for_type(df, model, CCP2)
        assert np.all(id_order1 == id_order2), "Per-type id orders must match."
        id_order = id_order1
        Z_aligned = np.vstack([Z[id_to_pos[pid], :] for pid in id_order])

        p1_prior = sigmoid(Z_aligned @ eta)
        m = np.maximum(ell1_vec, ell2_vec)
        w1 = p1_prior * np.exp(ell1_vec - m)
        w2 = (1.0 - p1_prior) * np.exp(ell2_vec - m)
        den = w1 + w2
        omega1 = w1 / np.maximum(den, 1e-300)
        omega2 = 1.0 - omega1

        def neg_logit_loss(eta_vec):
            p = sigmoid(Z_aligned @ eta_vec)
            eps = 1e-12
            return -np.sum(omega1 * np.log(np.maximum(p, eps)) +
                           omega2 * np.log(np.maximum(1 - p, eps)))

        eta = minimize(neg_logit_loss, eta, method='BFGS',
                       options=dict(maxiter=200, gtol=1e-5)).x

        def neg_wll_type(th, weights):
            spec = SpecType(beta=beta, sigma=sigma, **base_const,
                            gamma0=th[0], gamma1=th[1], delta0=th[2], delta1=th[3])
            _, CCP, _, _ = model.value_iteration_one_type(spec)
            ells, _ = per_id_loglike_for_type(df, model, CCP)
            return -np.sum(weights * ells)

        theta1 = minimize(lambda th: neg_wll_type(th, omega1), theta1,
                          method='Nelder-Mead',
                          options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3, disp=False)).x
        theta2 = minimize(lambda th: neg_wll_type(th, omega2), theta2,
                          method='Nelder-Mead',
                          options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3, disp=False)).x

        def neg_wll_shared(bs):
            b, s = pack_shared(bs)
            spec1s = SpecType(beta=b, sigma=s, **base_const,
                              gamma0=theta1[0], gamma1=theta1[1],
                              delta0=theta1[2], delta1=theta1[3])
            spec2s = SpecType(beta=b, sigma=s, **base_const,
                              gamma0=theta2[0], gamma1=theta2[1],
                              delta0=theta2[2], delta1=theta2[3])
            _, CCP1s, _, _ = model.value_iteration_one_type(spec1s)
            _, CCP2s, _, _ = model.value_iteration_one_type(spec2s)
            ell1, _ = per_id_loglike_for_type(df, model, CCP1s)
            ell2, _ = per_id_loglike_for_type(df, model, CCP2s)
            p1 = sigmoid(Z_aligned @ eta)
            mix = p1 * np.exp(ell1) + (1 - p1) * np.exp(ell2)
            return -np.sum(np.log(np.maximum(mix, 1e-300)))

        beta_sigma_raw = minimize(neg_wll_shared, beta_sigma_raw, method='Nelder-Mead',
                                  options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3, disp=False)).x

    beta, sigma = pack_shared(beta_sigma_raw)
    base = dict(beta=beta, sigma=sigma, **base_const)
    spec1 = SpecType(**base, gamma0=theta1[0], gamma1=theta1[1], delta0=theta1[2], delta1=theta1[3])
    spec2 = SpecType(**base, gamma0=theta2[0], gamma1=theta2[1], delta0=theta2[2], delta1=theta2[3])

    V1, CCP1, polx1, pols1 = model.value_iteration_one_type(spec1)
    V2, CCP2, polx2, pols2 = model.value_iteration_one_type(spec2)

    avg_p1 = float(sigmoid((Z @ eta)).mean())

    est_df = pd.DataFrame({
        "param": ["beta", "sigma",
                  "g0_1", "g1_1", "d0_1", "d1_1",
                  "g0_2", "g1_2", "d0_2", "d1_2"]
                 + [f"eta_{k}" for k in range(Z.shape[1])] + ["avg_p1"],
        "estimate": [beta, sigma,
                     theta1[0], theta1[1], theta1[2], theta1[3],
                     theta2[0], theta2[1], theta2[2], theta2[3]]
                    + list(eta) + [avg_p1]
    })
    est_df.to_csv(os.path.join(outdir, "estimates.csv"), index=False)

    ell1_vec, id_order = per_id_loglike_for_type(df, model, CCP1)
    ell2_vec, _        = per_id_loglike_for_type(df, model, CCP2)
    Z_aligned = np.vstack([Z[id_to_pos[pid], :] for pid in id_order])
    p1_aligned = sigmoid(Z_aligned @ eta)
    m = np.maximum(ell1_vec, ell2_vec)
    w1 = p1_aligned * np.exp(ell1_vec - m)
    w2 = (1 - p1_aligned) * np.exp(ell2_vec - m)
    den = w1 + w2
    omega1 = w1 / np.maximum(den, 1e-300)
    omega2 = 1.0 - omega1
    resp = pd.DataFrame({
        "id": id_order,
        "omega_type1": omega1,
        "omega_type2": omega2,
        "p1_prior_aligned": p1_aligned
    })
    feat = pd.DataFrame(Z, columns=["const","age0","ln_a0"]); feat["id"] = ids
    resp = resp.merge(feat, on="id", how="left")
    resp.to_csv(os.path.join(outdir, "responsibilities.csv"), index=False)

    np.save(os.path.join(outdir,"V_type1.npy"), V1)
    np.save(os.path.join(outdir,"V_type2.npy"), V2)
    np.save(os.path.join(outdir,"CCP_type1.npy"), CCP1)
    np.save(os.path.join(outdir,"CCP_type2.npy"), CCP2)

    pol1, pol2 = [], []
    for gi, g in enumerate(age_grid):
        for ai, a in enumerate(a_grid):
            pol1.append([a, g, polx1[ai, gi], pols1[ai, gi]])
            pol2.append([a, g, polx2[ai, gi], pols2[ai, gi]])
    pd.DataFrame(pol1, columns=["a","age","x_star","s_star"]).to_csv(os.path.join(outdir,"policy_type1.csv"), index=False)
    pd.DataFrame(pol2, columns=["a","age","x_star","s_star"]).to_csv(os.path.join(outdir,"policy_type2.csv"), index=False)

    return dict(beta=beta, sigma=sigma, theta1=theta1, theta2=theta2, eta=eta, avg_p1=avg_p1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EM estimator with observable-dependent type priors for budget-embedded literacy costs.")
    parser.add_argument("--csv", type=str, required=True, help="Path to panel CSV with columns: id,t,a,y,x,age")
    parser.add_argument("--outdir", type=str, default="./outputs_em_obs", help="Directory to write outputs")
    parser.add_argument("--na", type=int, default=24, help="# asset grid points")
    parser.add_argument("--gh", type=int, default=5, help="Gauss–Hermite quadrature order")
    parser.add_argument("--em", type=int, default=6, help="# EM iterations")
    parser.add_argument("--nm", type=int, default=30, help="# Nelder–Mead steps per block update")
    args = parser.parse_args()

    res = run_em_obs(args.csv, outdir=args.outdir, n_a=args.na, gh_order=args.gh, em_iters=args.em, nm_steps=args.nm)
    print("Done. avg p(type1) =", res["avg_p1"])