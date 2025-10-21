
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from nfxp_model import NFXPFinLitConsBudgetAgeType, SpecType

# ---------- Data ----------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[['id','t','a','y','x','age']].copy().sort_values(['id','t']).reset_index(drop=True)
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

# ---------- Utilities ----------
def per_id_loglike_for_type(df, model, CCP_type):
    """Return vector ell_i (loglike per id) for a given type's CCPs."""
    ells = []
    for pid, sub in df.groupby('id'):
        ccp_x = model.ccp_x_marginal(sub['a'].to_numpy(), sub['age'].to_numpy(), CCP_type)
        x_idx = np.array([np.argmin(np.abs(model.xgrid - v)) for v in sub['x']])
        p = ccp_x[np.arange(len(x_idx)), x_idx]
        ells.append(np.sum(np.log(np.maximum(p, 1e-12))))
    return np.asarray(ells), df['id'].unique()

# ---------- EM ----------
def run_em(csv_path: str, outdir: str = "outputs_em", n_a: int = 24, gh_order: int = 5,
           em_iters: int = 10, nm_steps: int = 50):
    os.makedirs(outdir, exist_ok=True)
    df = load_data(csv_path)
    a_grid, age_grid, x_grid, s_grid, y_bar = prep_grids(df, n_a=n_a)
    model = NFXPFinLitConsBudgetAgeType(a_grid, age_grid, y_bar, x_grid, s_grid, gh_order=gh_order)

    # Initial params
    beta_sigma_raw = np.array([0.0, 0.0])            # shared (beta, sigma)
    theta1 = np.array([ 2.0, -0.2, -2.0, 0.2])       # (g0,g1,d0,d1) type 1
    theta2 = np.array([ 1.0,  0.1, -1.5, 0.1])       # type 2
    pi = np.array([0.5, 0.5])                        # mixing proportions

    base_const = dict(R_f=1.02, mu_lnR=np.log(1.06), sigma_lnR=0.20)

    # Prepare id index order for responsibilities
    ids = df['id'].unique()
    N = len(ids)

    def pack_shared(bs):
        beta  = 0.5 + 0.49 * (1/(1+np.exp(-bs[0])))
        sigma = 0.5 + 4.5  * (1/(1+np.exp(-bs[1])))
        return beta, sigma

    for em in range(em_iters):
        beta, sigma = pack_shared(beta_sigma_raw)

        # Solve DP for each type
        spec1 = SpecType(beta=beta, sigma=sigma, **base_const,
                         gamma0=theta1[0], gamma1=theta1[1], delta0=theta1[2], delta1=theta1[3])
        spec2 = SpecType(beta=beta, sigma=sigma, **base_const,
                         gamma0=theta2[0], gamma1=theta2[1], delta0=theta2[2], delta1=theta2[3])
        V1, CCP1, _, _ = model.value_iteration_one_type(spec1)
        V2, CCP2, _, _ = model.value_iteration_one_type(spec2)

        # --- E-step ---
        ell1_vec, id_order1 = per_id_loglike_for_type(df, model, CCP1)
        ell2_vec, id_order2 = per_id_loglike_for_type(df, model, CCP2)
        # ensure same id order
        assert np.all(id_order1 == id_order2)
        m = np.maximum(ell1_vec, ell2_vec)  # stabilization
        w1 = pi[0] * np.exp(ell1_vec - m)
        w2 = pi[1] * np.exp(ell2_vec - m)
        den = w1 + w2
        omega1 = w1 / np.maximum(den, 1e-300)   # responsibilities for type 1
        omega2 = 1.0 - omega1

        # --- M-step: update mixing weights ---
        pi = np.array([omega1.mean(), omega2.mean()])
        pi = pi / pi.sum()

        # --- M-step: weighted re-fit of type params ---
        # Precompute per-id index splits to speed up objective evaluation
        groups = [g.index.values for _, g in df.groupby('id')]

        def neg_wll_type1(th):
            spec = SpecType(beta=beta, sigma=sigma, **base_const,
                            gamma0=th[0], gamma1=th[1], delta0=th[2], delta1=th[3])
            _, CCP, _, _ = model.value_iteration_one_type(spec)
            ells, _ = per_id_loglike_for_type(df, model, CCP)
            return -np.sum(omega1 * ells)

        def neg_wll_type2(th):
            spec = SpecType(beta=beta, sigma=sigma, **base_const,
                            gamma0=th[0], gamma1=th[1], delta0=th[2], delta1=th[3])
            _, CCP, _, _ = model.value_iteration_one_type(spec)
            ells, _ = per_id_loglike_for_type(df, model, CCP)
            return -np.sum(omega2 * ells)

        theta1 = minimize(neg_wll_type1, theta1, method='Nelder-Mead',
                          options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3, disp=False)).x
        theta2 = minimize(neg_wll_type2, theta2, method='Nelder-Mead',
                          options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3, disp=False)).x

        # --- Optional ECM step: update shared (beta, sigma) ---
        def neg_wll_shared(bs):
            b, s = pack_shared(bs)
            spec1s = SpecType(beta=b, sigma=s, **base_const,
                              gamma0=theta1[0], gamma1=theta1[1], delta0=theta1[2], delta1=theta1[3])
            spec2s = SpecType(beta=b, sigma=s, **base_const,
                              gamma0=theta2[0], gamma1=theta2[1], delta0=theta2[2], delta1=theta2[3])
            _, CCP1s, _, _ = model.value_iteration_one_type(spec1s)
            _, CCP2s, _, _ = model.value_iteration_one_type(spec2s)
            ell1, _ = per_id_loglike_for_type(df, model, CCP1s)
            ell2, _ = per_id_loglike_for_type(df, model, CCP2s)
            mix = pi[0]*np.exp(ell1) + pi[1]*np.exp(ell2)
            return -np.sum(np.log(np.maximum(mix, 1e-300)))
        beta_sigma_raw = minimize(neg_wll_shared, beta_sigma_raw, method='Nelder-Mead',
                                  options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3, disp=False)).x

    # Final solve with estimates
    beta, sigma = pack_shared(beta_sigma_raw)
    spec1 = SpecType(beta=beta, sigma=sigma, **base_const,
                     gamma0=theta1[0], gamma1=theta1[1], delta0=theta1[2], delta1=theta1[3])
    spec2 = SpecType(beta=beta, sigma=sigma, **base_const,
                     gamma0=theta2[0], gamma1=theta2[1], delta0=theta2[2], delta1=theta2[3])
    V1, CCP1, polx1, pols1 = model.value_iteration_one_type(spec1)
    V2, CCP2, polx2, pols2 = model.value_iteration_one_type(spec2)

    # Save
    pd.DataFrame({
        "param":["beta","sigma","g0_1","g1_1","d0_1","d1_1","g0_2","g1_2","d0_2","d1_2","pi1","pi2"],
        "estimate":[beta,sigma,theta1[0],theta1[1],theta1[2],theta1[3],theta2[0],theta2[1],theta2[2],theta2[3],pi[0],pi[1]]
    }).to_csv(os.path.join(outdir,"estimates.csv"), index=False)

    np.save(os.path.join(outdir,"V_type1.npy"), V1)
    np.save(os.path.join(outdir,"V_type2.npy"), V2)
    np.save(os.path.join(outdir,"CCP_type1.npy"), CCP1)
    np.save(os.path.join(outdir,"CCP_type2.npy"), CCP2)

    # Responsibilities per id (last E-step)
    omega = pd.DataFrame({"id": ids, "omega_type1": omega1, "omega_type2": omega2})
    omega.to_csv(os.path.join(outdir,"responsibilities.csv"), index=False)

    # Policies
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

    return dict(beta=beta, sigma=sigma, theta1=theta1, theta2=theta2, pi=pi)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="./outputs_em")
    p.add_argument("--na", type=int, default=24)
    p.add_argument("--gh", type=int, default=5)
    p.add_argument("--em", type=int, default=10)
    p.add_argument("--nm", type=int, default=50)
    args = p.parse_args()

    res = run_em(args.csv, outdir=args.outdir, n_a=args.na, gh_order=args.gh, em_iters=args.em, nm_steps=args.nm)
    print("Done. pi =", res["pi"])
