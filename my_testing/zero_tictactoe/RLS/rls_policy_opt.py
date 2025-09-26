
# rls_policy_opt.py
# Recursive Lexicographical Search (RLS) to optimize θ for Zero Tic‑Tac‑Toe.
# Objective: average payoff vs a perfect (minimax) opponent.

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

import zero_ttt_core as Z
from zero_ttt_minimax import MinimaxSolver

# ----- Feature indices (14-dim) -----
FEAT_DIM = 14
IDX = {
    "center":0, "corners":1, "sides":2,
    "two_open":3, "neg_opp_two_open":4,
    "inv3_diff":5, "inv2_diff":6, "inv1_diff":7,
    "ow_lt1":8, "ow_lt2":9, "ow_lt3":10,
    "secure_cells":11,
    "win_now":12, "need_block":13
}

# ----- Build θ from a subset -----
def theta_from_subset(values: List[float], indices: List[int], base: Optional[np.ndarray]=None) -> np.ndarray:
    th = np.zeros(FEAT_DIM, dtype=float) if base is None else np.array(base, dtype=float)
    assert len(th)==FEAT_DIM
    for v, j in zip(values, indices):
        th[j] = float(v)
    return th

# ----- Policy induced by θ (depth-limited search with Eθ leaves) -----
def theta_move(s: Z.State, theta: np.ndarray, depth: int=3):
    _, mv = Z.search_theta(s, depth=depth, theta=theta)
    if mv is None:
        L = Z.legal_moves(s)
        return None if not L else L[0]
    return mv

def play_vs_minimax(theta: np.ndarray, theta_side: int, solver: MinimaxSolver, depth_theta: int=3) -> int:
    """Return payoff from theta player's perspective: +1 win, 0 draw, -1 loss."""
    s = Z.initial_state()
    while True:
        term, w = Z.is_terminal(s.board)
        no_moves = (len(Z.legal_moves(s))==0) or (sum(s.invX)==0 and sum(s.invO)==0)
        if term or no_moves:
            if term:
                return +1 if w==theta_side else -1 if w==Z.other(theta_side) else 0
            return 0
        if s.to_move == theta_side:
            mv = theta_move(s, theta, depth=depth_theta)
            if mv is None: return 0
            s = Z.apply_move(s, mv)
        else:
            mv = solver.best_action(s)
            if mv is None: return 0
            s = Z.apply_move(s, mv)

def evaluate_theta(theta: np.ndarray, solver: MinimaxSolver, depth_theta: int=3) -> float:
    return 0.5*(play_vs_minimax(theta, Z.X, solver, depth_theta) + play_vs_minimax(theta, Z.O, solver, depth_theta))

# ----- RLS (recursive lexicographical search) -----
@dataclass
class Axis:
    center: float
    span: float
    low: float
    high: float

def rls_optimize(indices: List[int],
                 bounds: List[Tuple[float,float]],
                 levels: int = 4,
                 grid_sizes: List[int] = [11, 9, 7, 5],
                 rho: float = 0.35,
                 depth_theta: int = 3,
                 base_theta: Optional[np.ndarray]=None,
                 verbose: bool=True,
                 seed: int=0):
    """
    Optimize θ[indices] within given bounds using coordinate-wise recursive grid line searches.
    Returns: (theta_best, best_value, history)
    """
    assert len(indices)==len(bounds) > 0
    rng = np.random.default_rng(seed)
    # Init axes
    axes = []
    for (lo,hi) in bounds:
        c = 0.5*(lo+hi)
        s = 0.5*(hi-lo)
        axes.append(Axis(center=c, span=s, low=lo, high=hi))

    solver = MinimaxSolver()  # caches during all evaluations

    def vec_from_axes():
        return [ax.center for ax in axes]

    def evaluate_vec(vals):
        th = theta_from_subset(vals, indices, base=base_theta)
        return evaluate_theta(th, solver, depth_theta)

    best_vals = vec_from_axes()
    best_score = evaluate_vec(best_vals)
    best_th = theta_from_subset(best_vals, indices, base=base_theta)

    history = [("init", best_vals, best_score)]

    for L in range(levels):
        G = grid_sizes[min(L, len(grid_sizes)-1)]
        for j in range(len(indices)):
            ax = axes[j]
            # grid along axis j
            xs = np.linspace(max(ax.low, ax.center-ax.span), min(ax.high, ax.center+ax.span), G)
            best_local_x = ax.center
            best_local_sc = -9.0
            for x in xs:
                vals = vec_from_axes()
                vals[j] = float(x)
                sc = evaluate_vec(vals)
                if sc > best_local_sc:
                    best_local_sc = float(sc)
                    best_local_x = float(x)
            # update axis center to best x, shrink span
            ax.center = best_local_x
            ax.span = max(1e-3, ax.span * rho)
            # track global best
            if best_local_sc > best_score:
                best_score = best_local_sc
                best_vals = vec_from_axes()
                best_th = theta_from_subset(best_vals, indices, base=base_theta)
            if verbose:
                print(f"[L{L+1} axis {j+1}/{len(indices)}] center={ax.center:.3f}, span={ax.span:.3f}, best={best_score:+.3f}")
        history.append((f"level{L+1}", vec_from_axes(), best_score))

    return best_th, best_score, history

# ----- Convenience presets -----
def preset_indices(name: str) -> List[int]:
    if name=="win_block":
        return [IDX["win_now"], IDX["need_block"]]
    if name=="tactics4":
        return [IDX["win_now"], IDX["need_block"], IDX["two_open"], IDX["neg_opp_two_open"]]
    if name=="pos_tact6":
        return [IDX["win_now"], IDX["need_block"], IDX["two_open"], IDX["neg_opp_two_open"], IDX["corners"], IDX["secure_cells"]]
    raise ValueError("Unknown preset name.")

def default_bounds(indices: List[int]) -> List[Tuple[float,float]]:
    return [(-5.0, 5.0) for _ in indices]

# ----- Save / load -----
def save_theta(path: str, theta: np.ndarray):
    import json
    with open(path, 'w') as f:
        json.dump(theta.tolist(), f)
    return path

def load_theta(path: str) -> np.ndarray:
    import json
    with open(path, 'r') as f:
        arr = json.load(f)
    th = np.array(arr, dtype=float)
    if len(th) != FEAT_DIM:
        raise ValueError(f"Expected θ dim {FEAT_DIM}, got {len(th)}")
    return th
