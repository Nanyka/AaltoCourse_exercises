
#!/usr/bin/env python3
"""
run_estimation.py — generate data, estimate theta (Newton-Armijo), and print results.
"""
import numpy as np
from tictactoe_play_utils import (FEATURE_NAMES, features, logits_for_state,
                                  play_game, X, O)

# --- Likelihood, gradient, Hessian (multinomial logit across states) ---
def softmax(z):
    z = np.asarray(z, dtype=float)
    z -= np.max(z)
    ez = np.exp(z); s = ez.sum()
    return ez/s

def loglik(data, theta, lam=0.0):
    ll = 0.0
    for board, player, action in data:
        z, moves, Xmat = logits_for_state(board, player, theta)
        if not moves: continue
        p = softmax(z)
        j = moves.index(action)
        ll += np.log(max(p[j], 1e-12))
    return ll - 0.5*lam*np.dot(theta,theta)

def grad_hess(data, theta, lam=0.0):
    K = len(theta)
    g = np.zeros(K)
    H = np.zeros((K,K))
    for board, player, action in data:
        z, moves, Xmat = logits_for_state(board, player, theta)
        if not moves: continue
        p = softmax(z)
        j = moves.index(action)
        xj = Xmat[j]
        Ex = p @ Xmat
        g += xj - Ex
        Exx = (Xmat.T * p) @ Xmat
        H -= Exx - np.outer(Ex, Ex)
    g -= lam*theta
    H -= lam*np.eye(K)
    return g, H

def newton_maximize(data, theta0, lam=1e-6, max_iter=80, tol=1e-6, armijo=1e-4, backtrack=0.5, verbose=True):
    th = theta0.astype(float).copy()
    for it in range(1, max_iter+1):
        g, H = grad_hess(data, th, lam)
        gn = float(np.linalg.norm(g, 2))
        ll = loglik(data, th, lam)
        if verbose: print(f"iter {it:02d}: ll={ll:.3f} ||g||={gn:.3e} theta={th}")
        if gn < tol: break
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(-(H - 1e-6*np.eye(len(th))), g)
        t, base = 1.0, ll
        while True:
            th_new = th + t*step
            ll_new = loglik(data, th_new, lam)
            if ll_new >= base + armijo*t*float(g @ step):
                th = th_new; break
            t *= backtrack
            if t < 1e-8:
                th = th_new; break
    return th, loglik(data, th, lam)

# --- Data generation ---
def generate_dataset(n_games=1500, theta_star=None, seed=42):
    from tictactoe_play_utils import sample_policy_move, other, check_winner, apply_move
    if theta_star is None:
        theta_star = np.array([3.0, 2.0, 0.5, 0.3, 0.0, 0.6])
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_games):
        first = X if rng.random() < 0.5 else O
        board = tuple([0]*9); player = first
        while True:
            w, draw = check_winner(board)
            if w!=0 or draw: break
            mv = sample_policy_move(board, player, theta_star, rng)
            if mv is None: break
            data.append((board, player, mv))
            board = apply_move(board, mv, player)
            player = other(player)
    return data, theta_star

def save_csv(data, path):
    import csv
    def board_to_string(board): return ''.join('.XO'[v] for v in board)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["board","player","action"])
        for (b,p,a) in data: w.writerow([board_to_string(b), p, a])

def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1500)
    parser.add_argument("--theta", type=float, nargs=6, default=[3.0,2.0,0.5,0.3,0.0,0.6])
    parser.add_argument("--lam", type=float, default=1e-6, help="ridge regularization")
    parser.add_argument("--csv_out", type=str, default="tictactoe_dataset.csv")
    args = parser.parse_args()

    data, theta_star = generate_dataset(n_games=args.games, theta_star=np.array(args.theta))
    print(f"Generated {len(data)} decision points from {args.games} games.")
    save_csv(data, args.csv_out)
    print(f"Saved dataset to {args.csv_out}")

    theta0 = np.zeros(len(FEATURE_NAMES))
    theta_hat, ll = newton_maximize(data, theta0, lam=args.lam, verbose=True)
    print("\n=== Results ===")
    print("Feature names  :", FEATURE_NAMES)
    print("True theta_star:", theta_star)
    print("Estimated theta:", theta_hat)
    print("Log-likelihood :", ll)

if __name__ == "__main__":
    main()
