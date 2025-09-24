
#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
import argparse
from tictactoe_play_utils import (
    FEATURE_NAMES, logits_for_state,
    sample_policy_move, other, check_winner, apply_move,
    X, O
)

def softmax(z):
    z = np.asarray(z, dtype=float); z -= np.max(z); ez = np.exp(z); return ez/ez.sum()

def loglik(dataset, theta, lam=0.0):
    ll = 0.0
    for board, player, action in dataset:
        z, moves, Xmat = logits_for_state(board, player, theta)
        if not moves: continue
        p = softmax(z); j = moves.index(action)
        ll += np.log(max(p[j], 1e-12))
    return ll - 0.5*lam*np.dot(theta, theta)

def generate_dataset(n_games=1200, theta_star=None, seed=123):
    if theta_star is None: theta_star = np.array([3.0,2.0,0.5,0.3,0.0,0.6])
    rng = np.random.default_rng(seed); data=[]
    for _ in range(n_games):
        first = X if rng.random()<0.5 else O
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

@dataclass
class NRLSResult:
    theta: np.ndarray
    value: float
    evaluations: int

def _center(lo, hi): return 0.5*(lo+hi)

def _shrink(lo, hi, v, shrink):
    span = hi-lo; half=0.5*shrink*span
    nlo, nhi = v-half, v+half
    if nhi<=nlo:
        eps = max(1e-6, 1e-3*abs(v)+1e-6)
        nlo, nhi = v-eps, v+eps
    return float(nlo), float(nhi)

def nrls_maximize(f, bounds, levels=(5,7,9), topk=5, shrink=0.4, verbose=True):
    K = len(bounds)
    best_theta = np.array([_center(lo,hi) for lo,hi in bounds], dtype=float)
    best_val = f(best_theta); evals=1
    for L, m in enumerate(levels, start=1):
        if verbose: print(f"\n== Level {L}/{len(levels)} | grid={m}, topk={topk}, shrink={shrink} ==")
        prefix_boxes = [tuple(bounds)]
        prefix_vals  = [()]

        for d in range(K):
            if verbose: print(f"  - Dimension {d} / {K-1}")
            scored = []
            next_boxes, next_prefixes = [], []
            for pi, box in enumerate(prefix_boxes):
                lo, hi = box[d]
                grid = np.linspace(lo, hi, m)
                for v in grid:
                    theta_proxy = np.array([
                        (prefix_vals[pi][j] if j<len(prefix_vals[pi]) else _center(*box[j])) if j<d else
                        (v if j==d else _center(*box[j]))
                    for j in range(K)], dtype=float)
                    val = f(theta_proxy); evals += 1
                    if val > best_val: best_val, best_theta = val, theta_proxy.copy()
                    scored.append((val, pi, float(v)))
            if not scored: break
            scored.sort(key=lambda t: t[0], reverse=True)
            keep = scored[:min(topk, len(scored))]
            for val, pi, v in keep:
                box = list(prefix_boxes[pi])
                nlo, nhi = _shrink(*box[d], v, shrink)
                box[d] = (nlo, nhi)
                next_boxes.append(tuple(box))
                old = prefix_vals[pi] if len(prefix_vals)>0 else ()
                old = tuple(old[:d])
                next_prefixes.append(old + (v,))
            prefix_boxes, prefix_vals = next_boxes, next_prefixes

        for box in prefix_boxes:
            theta_c = np.array([_center(*box[j]) for j in range(K)], dtype=float)
            val = f(theta_c); evals += 1
            if val > best_val: best_val, best_theta = val, theta_c.copy()

        bounds = [ _shrink(lo, hi, best_theta[j], shrink) for j,(lo,hi) in enumerate(bounds) ]
        if verbose: print(f"  >> level best so far: val={best_val:.4f}, theta={best_theta}")

    return NRLSResult(theta=best_theta, value=best_val, evaluations=evals)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--theta", type=float, nargs=6, default=[3.0,2.0,0.5,0.3,0.0,0.6])
    ap.add_argument("--lam", type=float, default=1e-6)
    ap.add_argument("--levels", type=int, nargs="+", default=[5,7,9])
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--shrink", type=float, default=0.4)
    ap.add_argument("--lo", type=float, nargs=6, default=[-4,-4,-2,-2,-2,-2])
    ap.add_argument("--hi", type=float, nargs=6, default=[ 6, 6,  2,  2,  2,  2])
    args = ap.parse_args()

    data, theta_star = generate_dataset(n_games=args.games, theta_star=np.array(args.theta))

    def f_obj(theta): return loglik(data, np.asarray(theta), lam=args.lam)

    bounds = list(zip(args.lo, args.hi))
    res = nrls_maximize(f_obj, bounds, levels=tuple(args.levels), topk=args.topk, shrink=args.shrink, verbose=True)
    print("\n=== NRLS Results ===")
    print("Features      :", FEATURE_NAMES)
    print("True theta*   :", theta_star)
    print("Estimated theta:", res.theta)
    print("Log-likelihood:", res.value)
    print("Evaluations   :", res.evaluations)

if __name__ == "__main__":
    main()
