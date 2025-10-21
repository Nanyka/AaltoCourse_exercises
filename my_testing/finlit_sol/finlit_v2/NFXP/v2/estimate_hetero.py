
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from nfxp_model import NFXPFinLitConsBudgetStateFx, Spec

# ---------- Load data ----------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[['id','t','a','y','x','age']].copy()
    df = df.sort_values(['id','t']).reset_index(drop=True)
    return df

def prep_grids(df: pd.DataFrame, n_a: int = 36):
    A_MIN = max(1.0, df['a'].quantile(0.03))
    A_MAX = df['a'].quantile(0.97)
    a_grid = np.linspace(A_MIN, A_MAX, n_a)
    x_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_grid = np.array([0.6, 0.8, 0.95])
    y_bar = float(df['y'].mean())
    return a_grid, x_grid, s_grid, y_bar

def discretize_x(x, x_grid):
    x_grid = np.asarray(x_grid)
    return x_grid[np.argmin(np.abs(x_grid - x))]

# ---------- Likelihood ----------
class Likelihood:
    def __init__(self, df: pd.DataFrame, model: NFXPFinLitConsBudgetStateFx, x_grid: np.ndarray):
        self.df = df.copy()
        self.x_grid = x_grid
        self.df['x_disc'] = self.df['x'].apply(lambda v: discretize_x(v, x_grid))
        self.model = model

    def pack(self, theta_raw):
        # theta_raw = [beta_raw, sigma_raw, gamma0, gamma1, delta0, delta1]
        beta = 0.5 + 0.49 * (1 / (1 + np.exp(-theta_raw[0])))
        sigma = 0.5 + 4.5 * (1 / (1 + np.exp(-theta_raw[1])))
        gamma0 = theta_raw[2]
        gamma1 = theta_raw[3]
        delta0 = theta_raw[4]
        delta1 = theta_raw[5]
        return beta, sigma, gamma0, gamma1, delta0, delta1

    def neg_loglike(self, theta_raw):
        beta, sigma, gamma0, gamma1, delta0, delta1 = self.pack(theta_raw)
        spec = Spec(beta=beta, sigma=sigma, R_f=1.02,
                    mu_lnR=np.log(1.06), sigma_lnR=0.20,
                    gamma0=gamma0, gamma1=gamma1, delta0=delta0, delta1=delta1)

        V, Q, CCP, pol_x, pol_s = self.model.value_iteration(spec)
        ccp_x_obs = self.model.ccp_x_marginal(self.df['a'].to_numpy(), CCP)

        x_idx = np.array([np.argmin(np.abs(self.x_grid - xi)) for xi in self.df['x_disc'].to_numpy()])
        eps = 1e-12
        probs = ccp_x_obs[np.arange(len(x_idx)), x_idx]
        return -np.sum(np.log(np.maximum(probs, eps)))

# ---------- Runner ----------
def run_estimation(csv_path: str, outdir: str = "outputs_statefx", n_a: int = 36,
                   maxiter: int = 200, gh_order: int = 6):
    os.makedirs(outdir, exist_ok=True)
    df = load_data(csv_path)
    a_grid, x_grid, s_grid, y_bar = prep_grids(df, n_a=n_a)
    model = NFXPFinLitConsBudgetStateFx(a_grid, y_bar=y_bar, x_grid=x_grid, s_grid=s_grid, gh_order=gh_order)
    like = Likelihood(df, model, x_grid)

    theta0 = np.array([0.0, 0.0, 2.0, -0.2, -2.0, 0.2])  # reasonable starting values
    res = minimize(like.neg_loglike, theta0, method='Nelder-Mead',
                   options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3, disp=True))

    beta, sigma, gamma0, gamma1, delta0, delta1 = like.pack(res.x)
    spec_hat = Spec(beta, sigma, R_f=1.02, mu_lnR=np.log(1.06), sigma_lnR=0.20,
                    gamma0=gamma0, gamma1=gamma1, delta0=delta0, delta1=delta1)
    V_hat, Q_hat, CCP_hat, policy_x_hat, policy_s_hat = model.value_iteration(spec_hat)

    pd.DataFrame({
        "param":["beta","sigma","gamma0","gamma1","delta0","delta1"],
        "estimate":[beta, sigma, gamma0, gamma1, delta0, delta1]
    }).to_csv(os.path.join(outdir, "estimates.csv"), index=False)
    np.save(os.path.join(outdir, "V_hat.npy"), V_hat)
    np.save(os.path.join(outdir, "CCP_hat.npy"), CCP_hat)
    pd.DataFrame({"a_grid":a_grid, "policy_x":policy_x_hat, "policy_s":policy_s_hat}).to_csv(os.path.join(outdir, "policy.csv"), index=False)
    return res, (beta, sigma, gamma0, gamma1, delta0, delta1)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs_statefx")
    p.add_argument("--na", type=int, default=36)
    p.add_argument("--gh", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=200)
    args = p.parse_args()

    res, est = run_estimation(args.csv, outdir=args.outdir, n_a=args.na, maxiter=args.maxiter, gh_order=args.gh)
    print("Done. Estimates:", est)
