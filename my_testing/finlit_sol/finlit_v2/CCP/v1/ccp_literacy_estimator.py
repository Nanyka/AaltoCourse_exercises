#!/usr/bin/env python3
"""
CCP Literacy Estimator — Minimal, End‑to‑End Script (Model 1: CCP + CCS + EM)
-----------------------------------------------------------------------------
Given a panel with columns [id, t, a, y, age, x] where x ∈ [0,1] is the risky
asset share, estimate a latent “financial literacy” factor L that lowers
participation/maintenance costs in a dynamic portfolio model.

What it does
------------
1) Bins state s=(a,y,age) into quantile cells.
2) Estimates nonparametric CCPs  P(j|s)  over a grid of risky-share actions
   j ∈ {0, 0.1, …, 1.0} (j=0 = non-participation), with Dirichlet smoothing.
3) Estimates empirical transitions  p(s'|s,j)  from the panel.
4) Computes inclusive values IV(s) = -log P(0|s) and approximates continuation
   differences ΔEV(s;j,0) via Conditional Choice Simulation (CCS) using CCPs
   and transitions (no Bellman solve).
5) Estimates structural parameters θ = {k0, (k_L, φ_L)_L, (π_L)_L} by
   minimum distance on CCP log-odds plus a lightweight EM over latent types L.
6) Outputs parameter summary and per-person posterior literacy probabilities.

Usage
-----
python ccp_literacy_estimator.py --data your.csv \
  --id id --time t --a a --y y --age age --x x \
  --bins_a 12 --bins_y 10 --bins_age 8 --beta 0.96 \
  --H 4 --draws 80 --types 2 --alpha 5.0 --out_posteriors literacy_posteriors.csv

Notes
-----
- Start with small H and draws for speed; increase for accuracy.
- Assumes logit errors and time-invariant latent types.
- Returns are not modeled explicitly; dynamic effects are soaked up by ΔEV.
- This is a research skeleton: for publication-grade estimation, add bootstraps,
  richer belief/return blocks, and robustness checks.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd

# -----------------------
# Utilities
# -----------------------
EULER_GAMMA = 0.5772156649015328606

def quantile_bins(series: pd.Series, nbins: int):
    qs = np.linspace(0, 1, nbins + 1)
    cuts = series.quantile(qs).values.astype(float)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i-1]:
            cuts[i] = cuts[i-1] + 1e-9
    return cuts

def cut_to_bins(x: float, cuts: np.ndarray) -> int:
    # return bin index in 0..len(cuts)-2
    return int(np.clip(np.searchsorted(cuts, x, side="right") - 1, 0, len(cuts) - 2))

def dirichlet_smooth(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, float)
    prior = alpha / len(counts)
    return (counts + prior) / (counts.sum() + alpha)

def inclusive_value_from_ccp(P: np.ndarray) -> float:
    """IV(s) = log ∑_m exp(v_m) with v(j)-v(0)=logP(j)-logP(0) ⇒ IV = -log P(0|s)."""
    eps = 1e-12
    P = np.clip(np.asarray(P, float), eps, 1 - eps)
    return -float(np.log(P[0]))

def softmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v = v - np.max(v)
    ex = np.exp(v)
    return ex / ex.sum()

# -----------------------
# Config dataclass
# -----------------------
@dataclass
class CONFIG:
    A_BINS: int = 12
    Y_BINS: int = 10
    AGE_BINS: int = 8
    X_GRID: list = None      # risky-share grid including 0
    ALPHA: float = 5.0       # Dirichlet smoothing mass for CCPs
    H: int = 4               # CCS horizon
    DRAWS: int = 80          # CCS Monte Carlo paths per (s,j)
    L_TYPES: int = 2         # number of literacy types
    BETA: float = 0.96       # discount factor
    RANDOM_SEED: int = 123

    def __post_init__(self):
        if self.X_GRID is None:
            self.X_GRID = [0.0] + [i/10.0 for i in range(1, 11)]  # 0,0.1,...,1.0

# -----------------------
# Estimator class
# -----------------------
class CCPLiteracyEstimator:
    def __init__(self, cfg: CONFIG):
        self.cfg = cfg
        np.random.seed(cfg.RANDOM_SEED)

    # 1) Binning & discretization
    def fit_bins(self, df: pd.DataFrame, col_a: str, col_y: str, col_age: str):
        self.a_cuts = quantile_bins(df[col_a], self.cfg.A_BINS)
        self.y_cuts = quantile_bins(df[col_y], self.cfg.Y_BINS)
        self.age_cuts = quantile_bins(df[col_age], self.cfg.AGE_BINS)

    def discretize(self, df: pd.DataFrame, col_id: str, col_t: str, col_a: str, col_y: str, col_age: str, col_x: str):
        df = df.copy()
        df["ia"] = df[col_a].apply(lambda v: cut_to_bins(v, self.a_cuts))
        df["iy"] = df[col_y].apply(lambda v: cut_to_bins(v, self.y_cuts))
        df["iage"] = df[col_age].apply(lambda v: cut_to_bins(v, self.age_cuts))
        # map x to nearest grid action index j
        grid = np.array(self.cfg.X_GRID)
        def x_to_j(x):
            return int(np.argmin(np.abs(grid - float(x))))
        df["j"] = df[col_x].apply(x_to_j)
        # next-period bins (drop last obs per id)
        df = df.sort_values([col_id, col_t]).reset_index(drop=True)
        for col in ["ia","iy","iage","j"]:
            df[col+"_next"] = df.groupby(col_id)[col].shift(-1)
        df = df.dropna(subset=["ia_next","iy_next","iage_next"]).copy()
        for col in ["ia_next","iy_next","iage_next","j_next"]:
            df[col] = df[col].astype(int)
        self.df_binned = df
        return df

    # 2) CCPs P(j|s)
    def estimate_ccps(self):
        J = len(self.cfg.X_GRID)
        counts = defaultdict(lambda: np.zeros(J, dtype=float))
        for _, row in self.df_binned.iterrows():
            s = (int(row["ia"]), int(row["iy"]), int(row["iage"]))
            j = int(row["j"])
            counts[s][j] += 1.0
        self.ccp = {}
        self.state_counts = {}
        for s, cvec in counts.items():
            self.state_counts[s] = float(cvec.sum())
            self.ccp[s] = dirichlet_smooth(cvec, alpha=self.cfg.ALPHA)
        self.J = J
        self.states = list(self.ccp.keys())

    # 3) Transitions p(s'|s,j)
    def estimate_transitions(self):
        trans = defaultdict(lambda: defaultdict(float))
        for _, row in self.df_binned.iterrows():
            s = (int(row["ia"]), int(row["iy"]), int(row["iage"]))
            j = int(row["j"]) 
            sp = (int(row["ia_next"]), int(row["iy_next"]), int(row["iage_next"]))
            trans[(s,j)][sp] += 1.0
        self.trans = {}
        for key, d in trans.items():
            items = list(d.items())
            probs = np.array([v for (_, v) in items], float)
            probs = probs / probs.sum()
            self.trans[key] = ([sp for (sp, _) in items], probs)

    def draw_next_state(self, s, j):
        key = (s, j)
        if key not in self.trans:
            # fallback: mix over all observed j at this s
            cand = [(k, v) for (k,v) in self.trans.items() if k[0]==s]
            if not cand:
                return s
            sps, ps = [], []
            for (_, (sp_list, p_list)) in cand:
                sps += sp_list
                ps += list(p_list / len(cand))
            ps = np.array(ps, float); ps = ps/ps.sum()
            idx = np.random.choice(len(sps), p=ps)
            return sps[idx]
        sps, ps = self.trans[key]
        idx = np.random.choice(len(sps), p=ps)
        return sps[idx]

    # 4) CCS ΔEV(s;j,0)
    def delta_ev_ccs(self, s, j):
        beta = self.cfg.BETA
        H = self.cfg.H
        R = self.cfg.DRAWS

        def path_val(start_s, initial_j):
            accs = []
            for _ in range(R):
                s_curr = self.draw_next_state(start_s, initial_j)
                # t+1 contribution
                P1 = self.ccp.get(s_curr, None)
                iv = 0.0 if P1 is None else inclusive_value_from_ccp(P1)
                acc = (beta ** 1) * iv
                # continue for h=2..H
                for h in range(2, H+1):
                    P_curr = self.ccp.get(s_curr, None)
                    if P_curr is None:
                        j_draw = 0
                    else:
                        j_draw = int(np.random.choice(self.J, p=P_curr))
                    s_curr = self.draw_next_state(s_curr, j_draw)
                    P2 = self.ccp.get(s_curr, None)
                    iv2 = 0.0 if P2 is None else inclusive_value_from_ccp(P2)
                    acc += (beta ** h) * iv2
                accs.append(acc)
            return float(np.mean(accs)) if accs else 0.0

        return path_val(s, j) - path_val(s, 0)

    def precompute_delta_ev(self):
        self.delta_cache = {}
        for s in self.states:
            for j in range(1, self.J):  # j=0 is base
                self.delta_cache[(s,j)] = self.delta_ev_ccs(s, j)

    # 5) Minimum-distance objective on log-odds
    def _a_bin_midpoints(self):
        mids = []
        for b in range(len(self.a_cuts)-1):
            mids.append(0.5*(self.a_cuts[b] + self.a_cuts[b+1]))
        return np.array(mids, float)

    def objective(self, params):
        K = self.cfg.L_TYPES
        k0 = float(params[0])
        kL = np.array(params[1:1+K], float)
        phiL = np.array(params[1+K:1+2*K], float)
        raw_pi = np.array(params[1+2*K:1+3*K-1], float) if K>1 else np.array([], float)
        if K>1:
            pi = np.clip(raw_pi, 1e-6, 1.0)
            pi = np.append(pi, max(1e-6, 1.0 - pi.sum()))
            pi = pi / pi.sum()
        else:
            pi = np.array([1.0])

        a_mids = self._a_bin_midpoints()
        beta = self.cfg.BETA
        loss, wsum = 0.0, 0.0
        for s in self.states:
            ia, iy, iage = s
            P = self.ccp[s]
            a_mid = a_mids[ia]
            # expected cost over types
            cost_by_L = (k0 + kL + phiL * a_mid)  # shape (K,)
            exp_cost = float(np.sum(pi * cost_by_L))
            for j in range(1, self.J):
                logodds_emp = float(np.log(P[j]) - np.log(P[0]))
                delta_ev = self.delta_cache.get((s,j), 0.0)
                lam = -exp_cost + beta * delta_ev
                w = max(self.state_counts.get(s,1.0), 1.0)
                loss += (logodds_emp - lam)**2 * w
                wsum += w
        return loss / max(wsum, 1.0)

    # 6) A very simple optimizer (random search + small coordinate moves)
    def minimize_objective(self, init_params, iters=250, step=1.0, shrink=0.95):
        rng = np.random.default_rng(self.cfg.RANDOM_SEED)
        theta = np.array(init_params, float)
        best = self.objective(theta)
        for it in range(iters):
            cand = theta + rng.normal(0, step, size=len(theta))
            val = self.objective(cand)
            if val < best:
                theta, best = cand, val
            step *= shrink if (it % 10 == 0) else 1.0
        return theta, best

    # 7) EM over latent types (lightweight)
    def build_type_ccps(self, theta):
        """Return a dict: for each type L, a function that maps s -> predicted CCPs under type L."""
        K = self.cfg.L_TYPES
        k0 = float(theta[0])
        kL = np.array(theta[1:1+K], float)
        phiL = np.array(theta[1+K:1+2*K], float)
        # ΔEV is common across types and precomputed
        a_mids = self._a_bin_midpoints()
        beta = self.cfg.BETA

        def ccp_for_type(L):
            cache = {}
            def ccps_at_state(s):
                if s in cache:
                    return cache[s]
                ia,_,_ = s
                a_mid = a_mids[ia]
                lam = np.zeros(self.J)
                lam[0] = 0.0
                for j in range(1, self.J):
                    lam[j] = - (k0 + kL[L] + phiL[L]*a_mid) + beta * self.delta_cache.get((s,j), 0.0)
                p = softmax(lam)
                cache[s] = p
                return p
            return ccps_at_state
        return [ccp_for_type(L) for L in range(K)]

    def em(self, init_theta, n_iter=5):
        K = self.cfg.L_TYPES
        theta = np.array(init_theta, float)
        # initialize mixture weights π
        if K>1:
            raw_pi = theta[1+2*K:1+3*K-1]
            pi = np.clip(raw_pi, 1e-6, 1.0)
            pi = np.append(pi, max(1e-6, 1.0 - pi.sum()))
            pi = pi/pi.sum()
        else:
            pi = np.array([1.0])

        # build per-person sequences of (s_t, j_t)
        seqs = defaultdict(list)
        for _, row in self.df_binned.sort_values(["id","t"]).iterrows():
            s_t = (int(row["ia"]), int(row["iy"]), int(row["iage"]))
            j_t = int(row["j"])
            seqs[int(row["id"])].append((s_t, j_t))

        for it in range(n_iter):
            # E-step: responsibilities ω_{iL}
            type_ccps = self.build_type_ccps(theta)
            omega = {}  # id -> array(K)
            for pid, traj in seqs.items():
                like = np.zeros(K)
                for L in range(K):
                    ccps = type_ccps[L]
                    ll = 0.0
                    for (s_t, j_t) in traj:
                        P = ccps(s_t)
                        p = max(P[j_t], 1e-12)
                        ll += np.log(p)
                    like[L] = np.exp(ll)
                post = pi * like
                s = post.sum()
                if s<=0: post = np.ones(K)/K
                else: post = post/s
                omega[pid] = post
            # M-step (very light): update π and re-fit costs by MD with π fixed
            pi = np.mean(np.stack(list(omega.values())), axis=0)
            # pack back into theta (first K-1 mixture elements)
            if K>1:
                theta[1+2*K:1+3*K-1] = pi[:-1]
            # re-fit costs by MD with updated π
            theta, obj = self.minimize_objective(theta, iters=200, step=0.5, shrink=0.97)
        return theta, pi, omega

    # 8) Run full pipeline
    def run(self, df: pd.DataFrame, col_id: str, col_t: str, col_a: str, col_y: str, col_age: str, col_x: str,
            init_theta=None):
        # First stage
        self.fit_bins(df, col_a, col_y, col_age)
        dfb = self.discretize(df, col_id, col_t, col_a, col_y, col_age, col_x)
        # keep id,t for EM
        self.df_binned = self.df_binned.rename(columns={col_id:"id", col_t:"t"})
        self.estimate_ccps()
        self.estimate_transitions()
        self.precompute_delta_ev()
        # Second stage init
        K = self.cfg.L_TYPES
        if init_theta is None:
            init_theta = np.array([20.0] + [20.0]*K + [0.01]*K + ([1.0/K]*(K-1) if K>1 else []), float)
        # EM
        theta_hat, pi_hat, omega = self.em(init_theta, n_iter=4)
        self.theta_hat, self.pi_hat, self.omega = theta_hat, pi_hat, omega
        return theta_hat, pi_hat, omega

    # 9) Summaries & exports
    def summarize_params(self):
        K = self.cfg.L_TYPES
        th = np.array(self.theta_hat, float)
        k0 = float(th[0])
        kL = th[1:1+K].tolist()
        phiL = th[1+K:1+2*K].tolist()
        res = {"k0":k0, "kL":kL, "phiL":phiL, "pi": self.pi_hat.tolist()}
        return res

    def export_posteriors(self, out_csv: str):
        rows = []
        for pid, post in self.omega.items():
            rows.append({"id": pid, **{f"p_type{L}": float(post[L]) for L in range(len(post))}})
        pd.DataFrame(rows).to_csv(out_csv, index=False)

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="CCP Literacy Estimator (Model 1)")
    ap.add_argument('--data', required=True, help='Path to CSV with panel data')
    ap.add_argument('--id', default='id')
    ap.add_argument('--time', default='t')
    ap.add_argument('--a', default='a')
    ap.add_argument('--y', default='y')
    ap.add_argument('--age', default='age')
    ap.add_argument('--x', default='x')
    ap.add_argument('--bins_a', type=int, default=12)
    ap.add_argument('--bins_y', type=int, default=10)
    ap.add_argument('--bins_age', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--beta', type=float, default=0.96)
    ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--draws', type=int, default=80)
    ap.add_argument('--types', type=int, default=2)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out_posteriors', default='literacy_posteriors.csv')
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.data)
    req = [args.id, args.time, args.a, args.y, args.age, args.x]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Config
    cfg = CONFIG(
        A_BINS=args.bins_a, Y_BINS=args.bins_y, AGE_BINS=args.bins_age,
        ALPHA=args.alpha, H=args.H, DRAWS=args.draws, L_TYPES=args.types,
        BETA=args.beta, RANDOM_SEED=args.seed
    )
    est = CCPLiteracyEstimator(cfg)

    # Run
    theta_hat, pi_hat, omega = est.run(
        df=df,
        col_id=args.id, col_t=args.time, col_a=args.a, col_y=args.y, col_age=args.age, col_x=args.x
    )

    # Report
    print("\n=== Parameter summary ===")
    print(json.dumps(est.summarize_params(), indent=2))

    # Export posteriors
    est.export_posteriors(args.out_posteriors)
    print(f"\nSaved per-person literacy posteriors to: {args.out_posteriors}")

if __name__ == '__main__':
    main()
