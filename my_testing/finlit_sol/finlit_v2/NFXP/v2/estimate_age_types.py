
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from nfxp_model import NFXPFinLitConsBudgetAgeType, SpecType

# ---------- Load data ----------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[['id','t','a','y','x','age']].copy()
    df = df.sort_values(['id','t']).reset_index(drop=True)
    return df

def prep_grids(df: pd.DataFrame, n_a: int = 24):
    A_MIN = max(1.0, df['a'].quantile(0.05))
    A_MAX = df['a'].quantile(0.95)
    a_grid = np.linspace(A_MIN, A_MAX, n_a)
    age_grid = np.sort(df['age'].unique())
    x_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_grid = np.array([0.6, 0.8, 0.95])
    y_bar = float(df['y'].mean())
    return a_grid, age_grid, x_grid, s_grid, y_bar

def discretize_x(x, x_grid):
    x_grid = np.asarray(x_grid)
    return x_grid[np.argmin(np.abs(x_grid - x))]

# ---------- Likelihood with 2-type mixture ----------
class Likelihood2Type:
    def __init__(self, df, model, x_grid):
        self.df = df.copy()
        self.x_grid = x_grid
        self.df['x_disc'] = self.df['x'].apply(lambda v: discretize_x(v, x_grid))
        self.model = model

    def unpack(self, theta_raw):
        # theta = [beta_raw, sigma_raw,
        #          gamma0_1, gamma1_1, delta0_1, delta1_1,
        #          gamma0_2, gamma1_2, delta0_2, delta1_2,
        #          alpha]   # mixture logit for type 1; p1 = sigmoid(alpha), p2 = 1-p1
        beta = 0.5 + 0.49 * (1/(1+np.exp(-theta_raw[0])))
        sigma = 0.5 + 4.5 * (1/(1+np.exp(-theta_raw[1])))
        g0_1, g1_1, d0_1, d1_1 = theta_raw[2:6]
        g0_2, g1_2, d0_2, d1_2 = theta_raw[6:10]
        alpha = theta_raw[10]
        p1 = 1/(1+np.exp(-alpha))
        p2 = 1 - p1
        return beta, sigma, (g0_1,g1_1,d0_1,d1_1), (g0_2,g1_2,d0_2,d1_2), (p1,p2)

    def neg_ll(self, theta_raw):
        beta, sigma, par1, par2, (p1,p2) = self.unpack(theta_raw)
        # Build type specs
        base = dict(beta=beta, sigma=sigma, R_f=1.02, mu_lnR=np.log(1.06), sigma_lnR=0.20)
        spec1 = SpecType(**base, gamma0=par1[0], gamma1=par1[1], delta0=par1[2], delta1=par1[3])
        spec2 = SpecType(**base, gamma0=par2[0], gamma1=par2[1], delta0=par2[2], delta1=par2[3])

        # Solve DP for each type
        V1, CCP1, _, _ = self.model.value_iteration_one_type(spec1)
        V2, CCP2, _, _ = self.model.value_iteration_one_type(spec2)

        # For each observation, form P(x|a,age) as mixture over types (marginalizing s)
        ccp1_x = self.model.ccp_x_marginal(self.df['a'].to_numpy(), self.df['age'].to_numpy(), CCP1)
        ccp2_x = self.model.ccp_x_marginal(self.df['a'].to_numpy(), self.df['age'].to_numpy(), CCP2)
        # mixture
        mix = p1 * ccp1_x + p2 * ccp2_x

        x_idx = np.array([np.argmin(np.abs(self.x_grid - xi)) for xi in self.df['x_disc'].to_numpy()])
        eps = 1e-12
        probs = mix[np.arange(len(x_idx)), x_idx]
        return -np.sum(np.log(np.maximum(probs, eps)))

# ---------- Runner ----------
def run_estimation(csv_path: str, outdir="outputs_age_types", n_a=24, gh_order=5, maxiter=120):
    os.makedirs(outdir, exist_ok=True)
    df = load_data(csv_path)
    a_grid, age_grid, x_grid, s_grid, y_bar = prep_grids(df, n_a=n_a)

    model = NFXPFinLitConsBudgetAgeType(a_grid, age_grid, y_bar, x_grid, s_grid, gh_order=gh_order)
    like = Likelihood2Type(df, model, x_grid)

    theta0 = np.array([0.0, 0.0,   # beta_raw, sigma_raw
                       2.0, -0.2, -2.0, 0.2,   # type1
                       1.0,  0.1, -1.5, 0.1,   # type2
                       0.0])                   # alpha (mixture logit ~ 0.5)
    res = minimize(like.neg_ll, theta0, method='Nelder-Mead',
                   options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3, disp=True))

    # Unpack and compute outputs to save policies for type 1 (as example)
    beta, sigma, par1, par2, (p1,p2) = like.unpack(res.x)
    base = dict(beta=beta, sigma=sigma, R_f=1.02, mu_lnR=np.log(1.06), sigma_lnR=0.20)
    spec1 = SpecType(**base, gamma0=par1[0], gamma1=par1[1], delta0=par1[2], delta1=par1[3])
    spec2 = SpecType(**base, gamma0=par2[0], gamma1=par2[1], delta0=par2[2], delta1=par2[3])

    V1, CCP1, polx1, pols1 = model.value_iteration_one_type(spec1)
    V2, CCP2, polx2, pols2 = model.value_iteration_one_type(spec2)

    # Save
    import pandas as pd, numpy as np
    pd.DataFrame({
        "param":["beta","sigma","g0_1","g1_1","d0_1","d1_1","g0_2","g1_2","d0_2","d1_2","p1","p2"],
        "estimate":[beta,sigma,par1[0],par1[1],par1[2],par1[3],par2[0],par2[1],par2[2],par2[3],p1,p2]
    }).to_csv(os.path.join(outdir,"estimates.csv"), index=False)

    np.save(os.path.join(outdir,"V_type1.npy"), V1)
    np.save(os.path.join(outdir,"V_type2.npy"), V2)
    np.save(os.path.join(outdir,"CCP_type1.npy"), CCP1)
    np.save(os.path.join(outdir,"CCP_type2.npy"), CCP2)

    # Policies (save both types)
    pol1 = []
    for gi, g in enumerate(age_grid):
        for ai, a in enumerate(a_grid):
            pol1.append([a, g, polx1[ai, gi], pols1[ai, gi]])
    pd.DataFrame(pol1, columns=["a","age","x_star","s_star"]).to_csv(os.path.join(outdir,"policy_type1.csv"), index=False)

    pol2 = []
    for gi, g in enumerate(age_grid):
        for ai, a in enumerate(a_grid):
            pol2.append([a, g, polx2[ai, gi], pols2[ai, gi]])
    pd.DataFrame(pol2, columns=["a","age","x_star","s_star"]).to_csv(os.path.join(outdir,"policy_type2.csv"), index=False)

    return res

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs_age_types")
    p.add_argument("--na", type=int, default=24)
    p.add_argument("--gh", type=int, default=5)
    p.add_argument("--maxiter", type=int, default=120)
    args = p.parse_args()

    res = run_estimation(args.csv, outdir=args.outdir, n_a=args.na, gh_order=args.gh, maxiter=args.maxiter)
    print("Optimization success:", res.success, "nit:", res.nit)
