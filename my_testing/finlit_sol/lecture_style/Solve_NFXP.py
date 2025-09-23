#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve_NFXP.py
Full-solution MLE (Rust 1987) for the financial-literacy DDC using model_finlit.
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from model_finlit import (Spec, load_panel, build_state_action, transitions_by_observed,
                          expand_Pi_to_full, state_medians, flow_utility_matrix,
                          solve_bellman, build_obs_aggregator)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--Ka", type=int, default=5)
    ap.add_argument("--Ky", type=int, default=5)
    ap.add_argument("--KL", type=int, default=3)
    ap.add_argument("--Kage", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.96)
    ap.add_argument("--rf", type=float, default=0.01)
    ap.add_argument("--muR", type=float, default=0.05)
    ap.add_argument("--sdR", type=float, default=0.15)
    ap.add_argument("--sigma0", type=float, default=2.0)
    ap.add_argument("--save_grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--risk_grid", type=str, default="0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    spec = Spec(beta=args.beta, rf=args.rf, mu_R=args.muR, sd_R=args.sdR, sigma=args.sigma0,
                save_grid=tuple(float(x) for x in args.save_grid.split(",")),
                risk_grid=tuple(float(x) for x in args.risk_grid.split(",")))

    wide = load_panel(args.data)
    X, grids, obs_actions_pack, full_actions_pack = build_state_action(
        wide, spec, args.Ka, args.Ky, args.KL, args.Kage)
    obs_actions, obs_a_to_idx, obs_idx_to_a = obs_actions_pack
    full_actions, full_a_to_idx, full_idx_to_a = full_actions_pack

    states = sorted(X["state"].unique())
    S = len(states); D_full = len(full_actions); D_obs = len(obs_actions)
    Pi_obs = transitions_by_observed(X, states, D_obs)
    Pi = expand_Pi_to_full(Pi_obs, full_actions, obs_actions)
    y_s, a_s, L_s = state_medians(X, states)

    obs_to_full = build_obs_aggregator(full_actions, obs_actions)
    obs_i = np.array([states.index(s) for s in X["state"]])
    obs_a_obs = X["a_idx_obs"].values

    def neg_ll(par):
        log_sigma, F0, phi = par
        sigma = np.exp(log_sigma)
        U = flow_utility_matrix(S, full_actions, y_s, a_s, L_s,
                                F0, phi, sigma, spec.rf, spec.mu_R, spec.sd_R)
        V, CCP = solve_bellman(U, Pi, spec.beta)
        # aggregate CCP over latent savings s for observed choice (d,k)
        p = np.array([CCP[obs_i[t], obs_to_full[obs_a_obs[t]]].sum() for t in range(len(obs_i))]) + 1e-12
        return -np.sum(np.log(p))

    x0 = np.array([np.log(spec.sigma), 50.0, 10.0])
    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options=dict(maxiter=3000, maxfev=6000, xatol=1e-5, fatol=1e-5))

    log_sigma_hat, F0_hat, phi_hat = res.x
    sigma_hat = np.exp(log_sigma_hat)
    Uhat = flow_utility_matrix(S, full_actions, y_s, a_s, L_s,
                               F0_hat, phi_hat, sigma_hat, spec.rf, spec.mu_R, spec.sd_R)
    Vhat, CCPhat = solve_bellman(Uhat, Pi, spec.beta)

    # Save
    est = pd.DataFrame({"parameter":["sigma","F0","phi"], "estimate":[sigma_hat, F0_hat, phi_hat]})
    est.to_csv("finlit_estimates_nfxp.csv", index=False)
    np.save("V_hat_nfxp.npy", Vhat)
    np.save("CCP_hat_nfxp.npy", CCPhat)
    # policy on observed dimension: sum over savings choices and take argmax across (d,k)
    CCP_obs = np.zeros((S, len(obs_actions)))
    for j in range(len(obs_actions)):
        CCP_obs[:, j] = CCPhat[:, obs_to_full[j]].sum(axis=1)
    pol_obs = CCP_obs.argmax(axis=1)
    pol_df = pd.DataFrame({"state": states, "policy_obs_idx": pol_obs})
    pol_df.to_csv("policy_nfxp.csv", index=False)

    print("NFXP done:", est.to_dict(orient="list"))

if __name__ == "__main__":
    main()
