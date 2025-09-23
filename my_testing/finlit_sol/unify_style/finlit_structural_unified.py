#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified structural estimation for the financial-literacy DDC:
- NFXP (full-solution MLE, Rust 1987)
- NPL (Hotz–Miller pseudo-likelihood) with warm starts + Anderson acceleration
Guaranteed outputs in working dir:
  - finlit_estimates_<method>.csv
  - V_hat_<method>.npy
  - CCP_hat_<method>.npy
  - policy_<method>.csv
  - (for NPL) CCP_emp.npy
"""

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict
from scipy.optimize import minimize
from scipy.special import logsumexp

# ------------------ Utilities ------------------

def gh6():
    nodes = np.array([-2.350604973, -1.335849074, -0.436077412,
                       0.436077412,  1.335849074,  2.350604973])
    weights = np.array([0.0045300099, 0.15706732, 0.72462959,
                        0.72462959,   0.15706732, 0.0045300099])
    return nodes, weights

def quantile_edges(arr, K):
    qs = np.linspace(0, 1, K+1)
    edges = np.quantile(arr, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges

def digitize_safe(x, edges):
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges)-2)

def nearest_grid(val, grid):
    grid = np.asarray(grid, dtype=float)
    return float(grid[np.argmin(np.abs(grid - val))])

# ------------------ Spec ------------------

@dataclass
class ModelSpec:
    beta: float = 0.96
    rf: float = 0.01
    mu_R: float = 0.05
    sd_R: float = 0.15
    sigma: float = 2.0
    # use tuple (immutable) to avoid dataclass mutable default issues
    risk_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)

# ------------------ Data ------------------

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    wide = df.pivot_table(index="id", columns="wave", aggfunc="first")
    wide.columns = [f"{c}_{int(w)}" for c, w in wide.columns]
    wide = wide.reset_index().dropna()
    return wide

def build_state_action(wide: pd.DataFrame, spec: ModelSpec, Ka, Ky, KL, Kage):
    risk_grid = np.asarray(spec.risk_grid, dtype=float)
    X = pd.DataFrame({
        "id": wide["id"],
        "loga_1": np.log1p(wide["liquid_assets_eur_1"]),
        "logy_1": np.log(wide["income_eur_1"] + 1e-8),
        "L_1":    wide["literacy_index_z_1"],
        "age_1":  wide["age_1"],
        "d_1":    wide["participate_risky_1"].astype(int),
        "xshare_1": wide["risky_share_1"],
        "loga_2": np.log1p(wide["liquid_assets_eur_2"]),
        "logy_2": np.log(wide["income_eur_2"] + 1e-8),
        "L_2":    wide["literacy_index_z_2"],
        "age_2":  wide["age_2"],
        "d_2":    wide["participate_risky_2"].astype(int),
        "xshare_2": wide["risky_share_2"],
    })

    a_edges = quantile_edges(X["loga_1"].values, Ka)
    y_edges = quantile_edges(X["logy_1"].values, Ky)
    L_edges = quantile_edges(X["L_1"].values,   KL)
    age_edges = quantile_edges(X["age_1"].values, Kage)

    X["a_bin"]   = digitize_safe(X["loga_1"].values, a_edges)
    X["y_bin"]   = digitize_safe(X["logy_1"].values, y_edges)
    X["L_bin"]   = digitize_safe(X["L_1"].values,    L_edges)
    X["age_bin"] = digitize_safe(X["age_1"].values,  age_edges)
    X["state"]   = list(zip(X.a_bin, X.y_bin, X.L_bin, X.age_bin))

    X["a_bin2"]   = digitize_safe(X["loga_2"].values, a_edges)
    X["y_bin2"]   = digitize_safe(X["logy_2"].values, y_edges)
    X["L_bin2"]   = digitize_safe(X["L_2"].values,    L_edges)
    X["age_bin2"] = digitize_safe(X["age_2"].values,  age_edges)
    X["state2"]   = list(zip(X.a_bin2, X.y_bin2, X.L_bin2, X.age_bin2))

    actions = [(0, 0.0)] + [(1, k) for k in risk_grid]
    a_to_idx = {a:i for i, a in enumerate(actions)}
    idx_to_a = {i:a for a, i in a_to_idx.items()}

    X["xshare_1_disc"] = X["xshare_1"].apply(lambda v: nearest_grid(v, risk_grid))
    X["a_idx_obs"] = [a_to_idx[(int(d), (0.0 if int(d)==0 else float(xs)))]
                      for d, xs in zip(X["d_1"], X["xshare_1_disc"])]

    return X, actions, a_to_idx, idx_to_a, (a_edges, y_edges, L_edges, age_edges)

def transitions(X: pd.DataFrame, states, D):
    S = len(states); s_to_i = {s:i for i,s in enumerate(states)}
    Pi = np.ones((S, D, S))
    for _, r in X.iterrows():
        i = s_to_i[r["state"]]; a = r["a_idx_obs"]; j = s_to_i[r["state2"]]
        Pi[i, a, j] += 1.0
    Pi /= Pi.sum(axis=2, keepdims=True)
    return Pi

def state_medians(X: pd.DataFrame, states):
    s_to_i = {s:i for i,s in enumerate(states)}
    S = len(states)
    y_s = np.zeros(S); a_s = np.zeros(S); L_s = np.zeros(S)
    for s in states:
        sub = X.loc[X["state"]==s]
        i = s_to_i[s]
        if len(sub)==0:
            y_s[i] = np.exp(np.median(X["logy_1"]))
            a_s[i] = np.expm1(np.median(X["loga_1"]))
            L_s[i] = np.median(X["L_1"])
        else:
            y_s[i] = np.exp(np.median(sub["logy_1"]))
            a_s[i] = np.expm1(np.median(sub["loga_1"]))
            L_s[i] = np.median(sub["L_1"])
    return y_s, a_s, L_s

# ------------------ Preferences & flows ------------------

def ER_u(c_det, exposure, sigma, mu_R, sd_R):
    nodes, weights = gh6()
    vals = []
    for z, w in zip(nodes, weights):
        R = mu_R + sd_R * z/np.sqrt(2)
        c = np.maximum(c_det + exposure * (np.exp(R) - 1.0), 1e-8)
        if np.isclose(sigma, 1.0):
            u = np.log(c)
        else:
            u = np.power(c, 1.0 - sigma) / (1.0 - sigma)
        vals.append(w * u)
    return np.sum(vals, axis=0) / np.sqrt(np.pi)

def savings_rates_proxy(X: pd.DataFrame, D: int):
    res_avail = np.exp(X["logy_1"]) + np.expm1(X["loga_1"])
    srate = (np.expm1(X["loga_2"]).clip(0) / np.maximum(res_avail, 1e-8)).clip(0.0, 0.9)
    sr = X.assign(srate=srate).groupby("a_idx_obs")["srate"].mean()
    sr = sr.reindex(range(D)).fillna(0.1).values
    return sr

def flow_utility(S, D, idx_to_a, y_s, a_s, L_s, srate_by_a, F0, phi, sigma, rf, mu_R, sd_R):
    U = np.zeros((S, D))
    for i in range(S):
        y0, a0, L0 = y_s[i], a_s[i], L_s[i]
        for aidx in range(D):
            d, k = idx_to_a[aidx]
            cost = (F0 - phi * L0) if d != 0 else 0.0
            srate = srate_by_a[aidx]
            savings = srate * (y0 + a0)
            c_base = y0 + a0 * (1.0 + rf) - savings - cost
            c_det  = c_base + a0 * k * rf
            U[i, aidx] = ER_u(c_det, exposure=k*a0, sigma=sigma, mu_R=mu_R, sd_R=sd_R)
    return U

def solve_v(U, Pi, beta, tol=1e-6, maxit=500):
    S, D = U.shape
    V = np.zeros(S)
    for _ in range(maxit):
        EV = Pi @ V
        Q = U + beta * EV
        V_new = logsumexp(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    Q = U + beta * (Pi @ V)
    CCP = np.exp(Q - logsumexp(Q, axis=1, keepdims=True))
    return V, CCP

# ------------------ NFXP (full-solution MLE) ------------------

def nfxp_estimate(X, states, Pi, y_s, a_s, L_s, idx_to_a, spec: ModelSpec):
    obs_i = np.array([states.index(s) for s in X["state"]])
    obs_a = X["a_idx_obs"].values
    S = len(states); D = len(idx_to_a)

    srate_by_a = savings_rates_proxy(X, D)

    def neg_ll(par):
        log_sigma, F0, phi = par
        sigma = np.exp(log_sigma)
        U = flow_utility(S, D, idx_to_a, y_s, a_s, L_s, srate_by_a,
                         F0, phi, sigma, spec.rf, spec.mu_R, spec.sd_R)
        V, CCP = solve_v(U, Pi, spec.beta)
        p = CCP[obs_i, obs_a] + 1e-12
        return -np.sum(np.log(p))

    x0 = np.array([np.log(spec.sigma), 50.0, 10.0])
    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options=dict(maxiter=3000, maxfev=6000, xatol=1e-5, fatol=1e-5))
    log_sigma_hat, F0_hat, phi_hat = res.x
    sigma_hat = np.exp(log_sigma_hat)

    Uhat = flow_utility(S, D, idx_to_a, y_s, a_s, L_s, srate_by_a,
                        F0_hat, phi_hat, sigma_hat, spec.rf, spec.mu_R, spec.sd_R)
    Vhat, CCPhat = solve_v(Uhat, Pi, spec.beta)
    return {"opt": res, "sigma": sigma_hat, "F0": F0_hat, "phi": phi_hat,
            "U": Uhat, "V": Vhat, "CCP": CCPhat, "srate_by_a": srate_by_a}

# ------------------ Anderson acceleration for NPL ------------------

def anderson(F, x0, m=5, lam=1.0, maxit=1000, tol=1e-7):
    x = x0.copy()
    def res(v): return F(v) - v
    r = res(x)
    R = []  # residual history
    for k in range(1, maxit+1):
        R.append(r.reshape(-1,1))
        if len(R) > m: R.pop(0)
        if len(R) == 1:
            x_new = F(x)
        else:
            dR = np.hstack([R[i+1]-R[i] for i in range(len(R)-1)])
            try:
                gamma = np.linalg.lstsq(dR, r.reshape(-1,1), rcond=None)[0]
                dx = - (R[-1] - np.hstack(R[:-1]) @ gamma).ravel()
                x_trial = x + lam * dx
                Fx = F(x_trial)
                if not np.all(np.isfinite(Fx)):  # fallback
                    x_new = F(x)
                else:
                    x_new = Fx
            except np.linalg.LinAlgError:
                x_new = F(x)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k
        x = x_new; r = res(x)
    return x, maxit

# ------------------ NPL ------------------

def empirical_CCP(X, states, D):
    S = len(states)
    s_to_i = {s:i for i,s in enumerate(states)}
    CCP = np.ones((S, D))
    for _, r in X.iterrows():
        i = s_to_i[r["state"]]; a = r["a_idx_obs"]
        CCP[i, a] += 1.0
    CCP /= CCP.sum(axis=1, keepdims=True)
    return CCP

def solve_v_from_U(U, Pi, beta, V0=None, accel=True):
    S, D = U.shape
    if V0 is None: V0 = np.zeros(S)
    def T(V): return logsumexp(U + beta*(Pi @ V), axis=1)
    if accel:
        V, it = anderson(T, V0, m=5, lam=1.0, maxit=2000, tol=1e-8)
    else:
        V = V0.copy()
        for it in range(2000):
            V_new = T(V)
            if np.max(np.abs(V_new - V)) < 1e-8: V = V_new; break
            V = V_new
    Q = U + beta*(Pi @ V)
    CCP = np.exp(Q - logsumexp(Q, axis=1, keepdims=True))
    return V, CCP

def npl_estimate(X, states, Pi, y_s, a_s, L_s, idx_to_a, spec: ModelSpec,
                 sigma_fix: float = 2.0, start=(25.0, 10.0), accel=True):
    S = len(states); D = len(idx_to_a)
    obs_i = np.array([states.index(s) for s in X["state"]])
    obs_a = X["a_idx_obs"].values
    srate_by_a = savings_rates_proxy(X, D)
    CCP_emp = empirical_CCP(X, states, D)

    cache_V = np.zeros(S)

    def inner(F0, phi):
        U = flow_utility(S, D, idx_to_a, y_s, a_s, L_s, srate_by_a,
                         F0, phi, sigma_fix, spec.rf, spec.mu_R, spec.sd_R)
        nonlocal cache_V
        V, CCP = solve_v_from_U(U, Pi, spec.beta, V0=cache_V, accel=accel)
        cache_V = V
        return V, CCP, U

    def neg_pl(theta):
        F0, phi = theta
        V, CCP, U = inner(F0, phi)
        p = CCP[obs_i, obs_a] + 1e-12
        return -np.sum(np.log(p))

    res = minimize(neg_pl, np.array(start), method="Nelder-Mead",
                   options=dict(maxiter=2000, maxfev=4000, xatol=1e-5, fatol=1e-5))
    F0_hat, phi_hat = res.x
    V_hat, CCP_hat, U_hat = inner(F0_hat, phi_hat)
    return {"opt": res, "sigma_fix": sigma_fix, "F0": F0_hat, "phi": phi_hat,
            "V": V_hat, "CCP": CCP_hat, "U": U_hat, "CCP_emp": CCP_emp, "srate_by_a": srate_by_a}

# ------------------ Save helpers ------------------

def _ensure_ccp(out, Pi, beta):
    if "CCP" in out and isinstance(out["CCP"], np.ndarray):
        return out["CCP"]
    U, V = out["U"], out["V"]
    Q = U + beta * (Pi @ V)
    return np.exp(Q - logsumexp(Q, axis=1, keepdims=True))

def _save_all(method, out, states, Pi, beta, extra=None):
    suf = f"_{method}"
    CCP = _ensure_ccp(out, Pi, beta)
    np.save(f"V_hat{suf}.npy", out["V"])
    np.save(f"CCP_hat{suf}.npy", CCP)
    pol = CCP.argmax(axis=1)
    pd.DataFrame({"state": states, "policy_idx": pol}).to_csv(f"policy{suf}.csv", index=False)
    if extra:
        for name, arr in extra.items():
            np.save(name, arr)
    print("Wrote:", f"V_hat{suf}.npy", f"CCP_hat{suf}.npy", f"policy{suf}.csv")

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--method", type=str, default="nfxp", choices=["nfxp","npl"])
    ap.add_argument("--Ka", type=int, default=5)
    ap.add_argument("--Ky", type=int, default=5)
    ap.add_argument("--KL", type=int, default=3)
    ap.add_argument("--Kage", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.96)
    ap.add_argument("--rf", type=float, default=0.01)
    ap.add_argument("--muR", type=float, default=0.05)
    ap.add_argument("--sdR", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=2.0)  # NFXP uses; NPL fixes
    ap.add_argument("--F0_start", type=float, default=25.0)
    ap.add_argument("--phi_start", type=float, default=10.0)
    ap.add_argument("--no_accel", action="store_true", help="disable Anderson acceleration for NPL")
    args = ap.parse_args()

    spec = ModelSpec(beta=args.beta, rf=args.rf, mu_R=args.muR, sd_R=args.sdR, sigma=args.sigma)

    print("Loading panel…")
    wide = load_panel(args.data)
    X, actions, a_to_idx, idx_to_a, edges = build_state_action(wide, spec, args.Ka, args.Ky, args.KL, args.Kage)
    states = sorted(X["state"].unique())
    print(f"S={len(states)} states, D={len(actions)} actions")
    Pi = transitions(X, states, len(actions))
    y_s, a_s, L_s = state_medians(X, states)

    if args.method == "nfxp":
        print("Estimating by NFXP (full-solution MLE)…")
        out = nfxp_estimate(X, states, Pi, y_s, a_s, L_s, idx_to_a, spec)
        pd.DataFrame({"parameter":["sigma","F0","phi"],
                      "estimate":[out["sigma"], out["F0"], out["phi"]]}).to_csv("finlit_estimates_nfxp.csv", index=False)
        _save_all("nfxp", out, states, Pi, spec.beta)
        print(f"Done. sigma={out['sigma']:.4f}, F0={out['F0']:.4f}, phi={out['phi']:.4f}")
    else:
        print("Estimating by NPL (Hotz–Miller pseudo-LL) with warm starts + Anderson accel…")
        out = npl_estimate(X, states, Pi, y_s, a_s, L_s, idx_to_a, spec,
                           sigma_fix=args.sigma, start=(args.F0_start, args.phi_start), accel=(not args.no_accel))
        pd.DataFrame({"parameter":["sigma(fixed)","F0","phi"],
                      "estimate":[out["sigma_fix"], out["F0"], out["phi"]]}).to_csv("finlit_estimates_npl.csv", index=False)
        _save_all("npl", out, states, Pi, spec.beta, extra={"CCP_emp.npy": out["CCP_emp"]})
        print(f"Done. F0={out['F0']:.4f}, phi={out['phi']:.4f}, sigma(fixed)={out['sigma_fix']}")

    print("Outputs saved in current directory.")

if __name__ == "__main__":
    main()
