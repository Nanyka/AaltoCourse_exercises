#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_finlit.py
Core objects for the financial-literacy DDC in lecture notation.

State: x = (log a, log y, literacy L, age)
Controls (discretized):
  - participation d ∈ {0,1}
  - savings share s ∈ S = {s_1,...,s_Ks}  where a' = s * (y + a)
  - risky share k ∈ K = {0, k_2, ..., k_Kk} (only used when d=1; k=0 for d=0)

Budget and utility:
  c(x,d,s,k,R') = y + a[(1-k)(1+r_f) + k(1+R')] - s(y+a) - 1{d≠0} F(L)
  u(c;σ)        = c^(1-σ)/(1-σ)  (or log(c) if σ≈1)
  Flow utility used in the DDC is expected utility over R'.

Shocks:
  i.i.d. EV1 taste shocks per alternative ⇒ log-sum-exp Bellman operator.

Transitions:
  π(x'|x, d, k) estimated nonparametrically from the panel (wave 1→2) by observed action (d,k),
  then replicated across the savings choice s (which is not observed in the panel).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ---------------- Utilities ----------------

def gh6():
    """6-point Gauss–Hermite (physicists') nodes and weights."""
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

# ---------------- Spec ----------------

@dataclass
class Spec:
    beta: float = 0.96
    rf: float = 0.01
    mu_R: float = 0.05
    sd_R: float = 0.15
    sigma: float = 2.0
    risk_grid: Tuple[float,...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    save_grid: Tuple[float,...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # s in [0,1)

@dataclass
class Grids:
    Ka: int; Ky: int; KL: int; Kage: int
    a_edges: np.ndarray; y_edges: np.ndarray; L_edges: np.ndarray; age_edges: np.ndarray

# ---------------- Data prep ----------------

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    wide = df.pivot_table(index="id", columns="wave", aggfunc="first")
    wide.columns = [f"{c}_{int(w)}" for c, w in wide.columns]
    wide = wide.reset_index().dropna()
    return wide

def build_state_action(wide: pd.DataFrame, spec: Spec,
                       Ka=5, Ky=5, KL=3, Kage=3):
    """Build discretized state X and action sets (full and observed)."""
    risk_grid = np.asarray(spec.risk_grid, dtype=float)
    save_grid = np.asarray(spec.save_grid, dtype=float)

    # States
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

    # Observed actions (d,k): used for transitions and likelihood aggregation
    obs_actions = [(0, 0.0)] + [(1, k) for k in risk_grid]
    obs_a_to_idx = {a:i for i,a in enumerate(obs_actions)}
    obs_idx_to_a = {i:a for a,i in obs_a_to_idx.items()}
    X["xshare_1_disc"] = X["xshare_1"].apply(lambda v: nearest_grid(v, risk_grid))
    X["a_idx_obs"] = [obs_a_to_idx[(int(d), (0.0 if int(d)==0 else float(xs)))]
                      for d, xs in zip(X["d_1"], X["xshare_1_disc"])]

    # Full actions (d, s, k)
    full_actions = []
    for d in [0,1]:
        for s in save_grid:
            for k in (risk_grid if d==1 else [0.0]):
                full_actions.append((d, float(s), float(k)))
    full_a_to_idx = {a:i for i,a in enumerate(full_actions)}
    full_idx_to_a = {i:a for a,i in full_a_to_idx.items()}

    grids = Grids(Ka,Ky,KL,Kage,a_edges,y_edges,L_edges,age_edges)
    return X, grids, (obs_actions, obs_a_to_idx, obs_idx_to_a), (full_actions, full_a_to_idx, full_idx_to_a)

# ---------------- Transitions ----------------

def transitions_by_observed(X: pd.DataFrame, states, D_obs):
    """Estimate π(x'|x, observed_action) with Laplace smoothing; then normalized."""
    S = len(states); s_to_i = {s:i for i,s in enumerate(states)}
    Pi_obs = np.ones((S, D_obs, S))
    for _, r in X.iterrows():
        i = s_to_i[r["state"]]; a = r["a_idx_obs"]; j = s_to_i[r["state2"]]
        Pi_obs[i, a, j] += 1.0
    Pi_obs /= Pi_obs.sum(axis=2, keepdims=True)
    return Pi_obs

def expand_Pi_to_full(Pi_obs, full_actions, obs_actions):
    """Replicate π along savings dimension: π(x'|x,d,s,k) = π(x'|x,d,k)."""
    S, D_obs, _ = Pi_obs.shape
    D_full = len(full_actions)
    Pi = np.zeros((S, D_full, S))
    # map full action to observed action index
    obs_idx = {a:i for i,a in enumerate(obs_actions)}
    for j, a_full in enumerate(full_actions):
        d,s,k = a_full
        a_obs = (d, k if d==1 else 0.0)
        io = obs_idx[a_obs]
        Pi[:, j, :] = Pi_obs[:, io, :]
    return Pi

# ---------------- State representatives ----------------

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

# ---------------- Flow utility ----------------

def expected_u(c_det, exposure, sigma, mu_R, sd_R):
    """E_R[u(c_det + exposure*(exp(R')-1))]."""
    nodes, weights = gh6()
    acc = 0.0
    for z,w in zip(nodes, weights):
        R = mu_R + sd_R * z/np.sqrt(2)
        c = np.maximum(c_det + exposure * (np.exp(R) - 1.0), 1e-8)
        if np.isclose(sigma, 1.0):
            u = np.log(c)
        else:
            u = np.power(c, 1.0 - sigma) / (1.0 - sigma)
        acc += w * u
    return acc / np.sqrt(np.pi)

def flow_utility_matrix(S, full_actions, y_s, a_s, L_s,
                        F0, phi, sigma, rf, mu_R, sd_R):
    """Return U[i, a] over (state i, full action a=(d,s,k))."""
    U = np.zeros((S, len(full_actions)))
    for i in range(S):
        y0, a0, L0 = y_s[i], a_s[i], L_s[i]
        for aidx, (d,s,k) in enumerate(full_actions):
            cost = (F0 - phi * L0) if d!=0 else 0.0
            a_prime = s * (y0 + a0)     # discretized savings rule
            c_det = y0 + a0*((1-k)*(1+rf) + k*(1+rf)) - a_prime - cost  # rf baseline on risky too; stochastic part via exposure below
            U[i, aidx] = expected_u(c_det, exposure=a0*k, sigma=sigma, mu_R=mu_R, sd_R=sd_R)
    return U

# ---------------- Bellman ----------------

def solve_bellman(U, Pi, beta, tol=1e-6, maxit=1000):
    """Inclusive-value iteration for EV1 shocks."""
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

# ---------------- Likelihood helpers ----------------

def build_obs_aggregator(full_actions, obs_actions):
    """Return list obs_to_full[j] = list of full action indices that map to observed action j=(d,k)."""
    D_full = len(full_actions); D_obs = len(obs_actions)
    obs_idx = {a:i for i,a in enumerate(obs_actions)}
    obs_to_full = [[] for _ in range(D_obs)]
    for j, a_full in enumerate(full_actions):
        d,s,k = a_full
        a_obs = (d, k if d==1 else 0.0)
        io = obs_idx[a_obs]
        obs_to_full[io].append(j)
    return obs_to_full
