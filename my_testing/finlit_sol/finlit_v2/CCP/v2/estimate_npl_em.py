
# (enhanced) two-type CCP–NPL+EM estimator with log-sum over s, optional fixed (beta,sigma), damping, and multi-restarts helper.
import numpy as np, pandas as pd, os, json
from dataclasses import dataclass
from scipy.optimize import minimize
from typing import Tuple
from nfxp_model import NFXPFinLitConsBudgetAgeType, SpecType

def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def logsumexp_axis(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze()
def softmax_axis(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); e = np.exp(x - m)
    den = np.sum(e, axis=axis, keepdims=True); return e / np.maximum(den, 1e-300)

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    needed = {'id','t','a','y','x','age'}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}. Found {df.columns.tolist()}")
    return df[['id','t','a','y','x','age']].copy().sort_values(['id','t']).reset_index(drop=True)

def prep_grids(df, n_a=24):
    A_MIN = max(1.0, float(df['a'].quantile(0.05))); A_MAX = float(df['a'].quantile(0.95))
    if A_MAX <= A_MIN: A_MAX = A_MIN + 1.0
    a_grid = np.linspace(A_MIN, A_MAX, n_a)
    age_grid = np.sort(df['age'].unique())
    if len(age_grid) > 25:
        qs = np.linspace(0,1,25); age_grid = np.unique(np.round(np.quantile(df['age'], qs), 3))
    x_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0]); s_grid = np.array([0.6, 0.8, 0.95])
    y_bar = float(df['y'].mean()); return a_grid, age_grid, x_grid, s_grid, y_bar

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, sub in df.groupby('id'):
        sub = sub.sort_values('t')
        age0 = float(sub['age'].iloc[0])
        a0 = float(sub['a'].iloc[0])
        ln_a0 = np.log(max(a0, 1e-8))
        rows.append([pid, 1.0, age0, ln_a0])
    return pd.DataFrame(rows, columns=['id','const','age0','ln_a0']).set_index('id')

def npl_operator(model: NFXPFinLitConsBudgetAgeType, spec: SpecType, V_prev: np.ndarray):
    Qx = np.zeros((model.NA, model.NG, model.NX))
    for gi, g in enumerate(model.agegrid):
        next_gi = min(gi + 1, model.NG - 1)
        for ai, a in enumerate(model.agrid):
            for xi, x in enumerate(model.xgrid):
                vals = []
                for si, s in enumerate(model.sgrid):
                    ev = 0.0
                    for k, w in enumerate(model.z_w): # w is the quadrature weight for node k in the expectation over the risky return.
                        Rk = np.exp(spec.mu_lnR + spec.sigma_lnR * model.z_nodes[k])
                        gross = a * (x * Rk + (1.0 - x) * spec.R_f) + model.y_bar
                        if x > 1e-12:
                            kfix = np.exp(spec.gamma0 + spec.gamma1 * g)
                            tau  = 0.9 / (1.0 + np.exp(-(spec.delta0 + spec.delta1 * g))) # multiplied by 0.9 to guarantee net resources stay positive when someone invests in the risky asset
                            res  = (1.0 - tau) * gross - kfix
                        else:
                            res  = gross
                        res = max(res, 1e-12)
                        c = (1.0 - s) * res
                        a_next = s * res
                        aj = int(np.argmin(np.abs(model.agrid - a_next)))
                        cont = V_prev[aj, next_gi] # expected VF
                        if abs(spec.sigma - 1.0) < 1e-12: u = np.log(max(c, 1e-12))
                        else: u = (max(c, 1e-12)**(1.0 - spec.sigma) - 1.0) / (1.0 - spec.sigma)
                        ev += w * (u + spec.beta * cont) # u + spec.beta * cont = choice-specific VF
                    vals.append(ev)
                vals = np.array(vals)
                m = vals.max()
                Qx[ai, gi, xi] = m + np.log(np.exp(vals - m).sum()) # take logsum for saving rate
    V_new = logsumexp_axis(Qx, axis=2) # take logsum for risky share --> integrated VF (ex-ante), lack of the Euler-Mascheroni constant
    CCP_x = softmax_axis(Qx, axis=2)
    return V_new, CCP_x, Qx

def npl_solve(model, spec, max_iter=200, tol=1e-6):
    V = np.zeros((model.NA, model.NG))
    for it in range(max_iter):
        V_new, CCP_x_new, Qx = npl_operator(model, spec, V)
        diff = np.max(np.abs(V_new - V))
        V = V_new # update V for the next iteration
        if diff < tol: break
    V_final, CCP_x_final, Qx_final = npl_operator(model, spec, V); return V_final, CCP_x_final, Qx_final

def per_id_loglike_x(df, model, CCP_x):
    ells, ids = [], [] # ells: per-person log-likelihoods
    for pid, sub in df.groupby('id'):
        ai = np.array([np.argmin(np.abs(model.agrid - v)) for v in sub['a']])
        gi = np.array([np.argmin(np.abs(model.agegrid - v)) for v in sub['age']])
        xi = np.array([np.argmin(np.abs(model.xgrid - v)) for v in sub['x']])
        p = CCP_x[ai, gi, xi]
        ells.append(float(np.sum(np.log(np.maximum(p, 1e-12)))))
        ids.append(pid)
    return np.asarray(ells), np.asarray(ids)

@dataclass
class NPLShared:
    beta: float = 0.95; sigma: float = 2.0
    R_f: float = 1.02; mu_lnR: float = np.log(1.06); sigma_lnR: float = 0.20

def pack_spec(shared: NPLShared, theta):
    return SpecType(beta=shared.beta, sigma=shared.sigma, R_f=shared.R_f,
                    mu_lnR=shared.mu_lnR, sigma_lnR=shared.sigma_lnR,
                    gamma0=theta[0], gamma1=theta[1], delta0=theta[2], delta1=theta[3])

def estimate_npl_em(csv_path: str, outdir: str = "outputs_npl_em",
                    n_a: int = 24, gh_order: int = 5,
                    theta1_0: Tuple[float,float,float,float]=(1.0,-0.01,-1.0,0.01),
                    theta2_0: Tuple[float,float,float,float]=(2.0, 0.01,-2.0,0.02),
                    eta0: Tuple[float,float,float]=(0.0,0.0,0.0),
                    em_iters: int = 6, nm_steps: int = 30, tol: float = 1e-6, seed: int = 68,
                    fix_beta_sigma: bool = True, beta_fix: float = 0.93, sigma_fix: float = 2.2,
                    damping: float = 0.0):
    os.makedirs(outdir, exist_ok=True); rng = np.random.default_rng(seed)
    df = load_data(csv_path)
    a_grid, age_grid, x_grid, s_grid, y_bar = prep_grids(df, n_a=n_a)
    model = NFXPFinLitConsBudgetAgeType(a_grid, age_grid, y_bar, x_grid, s_grid, gh_order=gh_order)
    Zdf = build_features(df)
    ids = Zdf.index.to_numpy()
    Z = Zdf.to_numpy()
    id_to_pos = {pid:i for i,pid in enumerate(ids)}

    shared = NPLShared(beta=beta_fix, sigma=sigma_fix) if fix_beta_sigma else NPLShared()
    theta1 = np.array(theta1_0, dtype=float); theta2 = np.array(theta2_0, dtype=float); eta = np.array(eta0, dtype=float)
    obs_ll_hist = []; omega1_old = None

    for em in range(em_iters):

        # Run inner loop for converged CCP
        spec1 = pack_spec(shared, theta1); spec2 = pack_spec(shared, theta2)
        V1, CCPx1, _ = npl_solve(model, spec1, max_iter=200, tol=tol)
        V2, CCPx2, _ = npl_solve(model, spec2, max_iter=200, tol=tol)

        # Calculate mixed total likelihood
        ell1, id_order1 = per_id_loglike_x(df, model, CCPx1) # ell1: per-person log-likelihoods if type 1
        ell2, id_order2 = per_id_loglike_x(df, model, CCPx2) # ell2: per-person log-likelihoods if type 2
        assert np.all(id_order1 == id_order2)
        id_order = id_order1
        Z_aligned = np.vstack([Z[id_to_pos[pid], :] for pid in id_order])
        p1 = sigmoid(Z_aligned @ eta)
        mix = p1*np.exp(ell1) + (1.0-p1)*np.exp(ell2)
        obs_ll = float(np.sum(np.log(np.maximum(mix, 1e-300)))); obs_ll_hist.append(obs_ll)

        # Update estimated type probabilities
        m = np.maximum(ell1, ell2);
        w1 = p1*np.exp(ell1 - m);
        w2 = (1.0-p1)*np.exp(ell2 - m)

        den = w1 + w2;
        omega1 = w1/np.maximum(den,1e-300);
        omega2 = 1.0 - omega1
        if damping > 0 and omega1_old is not None:
            omega1 = (1-damping)*omega1_old + damping*omega1;
            omega2 = 1.0 - omega1
        omega1_old = omega1.copy()

        # we have to solve min neg_logit since scipy.optimize.minimize solve minimization problems
        def neg_logit(eta_vec):
            p = sigmoid(Z_aligned @ eta_vec); eps = 1e-12 # prior probability Pr(type=1∣Zi;η).
            return -np.sum(omega1*np.log(np.maximum(p,eps)) + omega2*np.log(np.maximum(1-p,eps)))
        eta = minimize(neg_logit, eta, method='BFGS', options=dict(maxiter=200, gtol=1e-5)).x

        def neg_wpll(th, weights):
            spec = pack_spec(shared, th)
            _, CCPx, _ = npl_solve(model, spec, max_iter=200, tol=tol)
            ells, _ = per_id_loglike_x(df, model, CCPx)
            return -np.sum(weights * ells)
        theta1 = minimize(lambda th: neg_wpll(th, omega1), theta1, method='Nelder-Mead',
                          options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3)).x
        theta2 = minimize(lambda th: neg_wpll(th, omega2), theta2, method='Nelder-Mead',
                          options=dict(maxiter=nm_steps, xatol=1e-3, fatol=1e-3)).x

    est = pd.DataFrame({"param":["g0_1","g1_1","d0_1","d1_1","g0_2","g1_2","d0_2","d1_2","eta_const","eta_age0","eta_ln_a0"],
                        "estimate": list(theta1)+list(theta2)+list(eta)})
    est.to_csv(os.path.join(outdir, "estimates_npl_em.csv"), index=False)
    return dict(theta1=theta1, theta2=theta2, eta=eta, avg_p1=float(sigmoid(Z @ eta).mean()),
                obs_ll=obs_ll_hist[-1], obs_ll_hist=obs_ll_hist)

def estimate_npl_em_with_restarts(csv_path: str, outdir: str = "outputs_npl_em",
                                  restarts: int = 10, seed: int = 123, **kwargs):
    rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        th1 = np.array([rng.normal(1.0,0.5), rng.normal(0.0,0.02), rng.normal(-1.0,0.5), rng.normal(0.0,0.02)])
        th2 = np.array([rng.normal(2.0,0.5), rng.normal(0.0,0.02), rng.normal(-2.0,0.5), rng.normal(0.0,0.02)])
        eta0 = rng.normal(0.0, 0.2, size=3)
        out = estimate_npl_em(csv_path, outdir=outdir, theta1_0=th1, theta2_0=th2, eta0=eta0,
                               seed=rng.integers(1e9), **kwargs)
        if (best is None) or (out['obs_ll'] > best['obs_ll']):
            best = out
            with open(os.path.join(outdir, 'best_restart.json'), 'w') as f:
                json.dump({'restart': int(r), 'obs_ll': float(out['obs_ll'])}, f)
    return best

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Two-type CCP–NPL with EM (log-sum over s, restarts, damping, optional fixed beta/sigma)")
    p.add_argument("--csv", type=str, required=True); p.add_argument("--outdir", type=str, default="./outputs_npl_em")
    p.add_argument("--na", type=int, default=24); p.add_argument("--gh", type=int, default=5)
    p.add_argument("--em", type=int, default=6); p.add_argument("--nm", type=int, default=30)
    p.add_argument("--fixbeta", action="store_true"); p.add_argument("--beta", type=float, default=0.93); p.add_argument("--sigma", type=float, default=2.2)
    p.add_argument("--damping", type=float, default=0.0)
    args = p.parse_args()
    out = estimate_npl_em(args.csv, outdir=args.outdir, n_a=args.na, gh_order=args.gh, em_iters=args.em, nm_steps=args.nm,
                          fix_beta_sigma=args.fixbeta, beta_fix=args.beta, sigma_fix=args.sigma, damping=args.damping)
    print("Done.", out)
