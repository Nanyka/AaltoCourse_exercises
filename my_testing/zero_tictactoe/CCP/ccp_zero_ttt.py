
# ccp_zero_ttt.py — CCP (Hotz–Miller) for Zero Tic‑Tac‑Toe (clean, consistent version)
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional
from functools import lru_cache

import zero_ttt_core as Z

# -------- Action id helpers --------
def action_id(v: int, i: int) -> int:
    return 3*i + (v-1)

def id_to_action(aid: int) -> Tuple[int,int]:
    i, r = divmod(aid, 3)
    return r+1, i

def legal_mask_27(s: Z.State) -> np.ndarray:
    mask = np.zeros(27, dtype=bool)
    for (v,i) in Z.legal_moves(s):
        mask[action_id(v,i)] = True
    return mask

# -------- D4 symmetries + canonical keys --------
def _perms():
    I  = [0,1,2,3,4,5,6,7,8]
    R  = [6,3,0,7,4,1,8,5,2]   # 90 CW
    R2 = [8,7,6,5,4,3,2,1,0]   # 180
    R3 = [2,5,8,1,4,7,0,3,6]   # 270
    H  = [2,1,0,5,4,3,8,7,6]   # vertical mirror
    V  = [6,7,8,3,4,5,0,1,2]   # horizontal mirror
    D  = [0,3,6,1,4,7,2,5,8]   # main diag
    A  = [8,5,2,7,4,1,6,3,0]   # anti diag
    return [I,R,R2,R3,H,V,D,A]

PERMS = _perms()

def apply_perm(board: Iterable[int], perm: List[int]) -> Tuple[int,...]:
    b = list(board)
    return tuple(b[p] for p in perm)

def canonical_key_board(s: Z.State) -> Tuple[int,...]:
    boards = [apply_perm(s.board, P) for P in PERMS]
    return min(boards)

def canonical_key_full(s: Z.State):
    # include inventories to avoid pooling states with different stock
    return (canonical_key_board(s), s.invX, s.invO)

# -------- Dataset --------
@dataclass
class Datum:
    state: Z.State
    a_id: int          # chosen action id (0..26), must be legal

def random_policy_move(s: Z.State, rng: np.random.Generator) -> Optional[Tuple[int,int]]:
    moves = Z.legal_moves(s)
    if not moves: return None
    return moves[int(rng.integers(0, len(moves)))]

def theta_policy_move(s: Z.State, theta: np.ndarray, depth: int=4) -> Optional[Tuple[int,int]]:
    val, mv = Z.search_theta(s, depth=depth, theta=np.asarray(theta, float))
    if mv is None:
        moves = Z.legal_moves(s)
        return None if not moves else moves[0]
    return mv

def generate_dataset(n_games: int=300, policy='random', theta: np.ndarray=None, depth: int=4, seed: int=0) -> List[Datum]:
    rng = np.random.default_rng(seed)
    data: List[Datum] = []
    for _ in range(n_games):
        s = Z.initial_state()
        while True:
            term, _ = Z.is_terminal(s.board)
            if term or (sum(s.invX)==0 and sum(s.invO)==0):
                break
            if policy=='theta' and theta is not None:
                mv = theta_policy_move(s, theta, depth=depth)
            else:
                mv = random_policy_move(s, rng)
            if mv is None: break
            data.append(Datum(s, action_id(mv[0], mv[1])))
            s = Z.apply_move(s, mv)
    return data

# -------- Empirical CCPs --------
def estimate_ccp(dataset: List[Datum], alpha: float=0.5):
    """
    Returns dict: ccp[(player, (canon_board, invX, invO))] -> 27-dim probs over legal.
    Laplace smoothing alpha on legal actions.
    """
    counts: Dict[Tuple[int, Tuple], np.ndarray] = {}
    masks : Dict[Tuple[int, Tuple], np.ndarray] = {}
    for d in dataset:
        key = (d.state.to_move, canonical_key_full(d.state))
        if key not in counts:
            counts[key] = np.zeros(27, dtype=float)
            masks[key]  = legal_mask_27(d.state)
        counts[key][d.a_id] += 1.0
    ccp = {}
    for key, c in counts.items():
        m = masks[key]
        c = c + alpha*m
        c[~m] = 0.0
        tot = c.sum()
        if tot <= 0:
            u = m.astype(float); u /= max(1.0, u.sum())
            ccp[key] = u
        else:
            ccp[key] = c / tot
    return ccp

# -------- Continuation offsets from CCP (finite horizon, memoized) --------
def EV_from_state(state: Z.State, p_reference: int, ccp: Dict, H: int=6, beta: float=1.0) -> float:
    """
    Expected terminal value for p_reference starting from 'state', following CCP.
    Returns in [-1,1]; H is the horizon (plies left). beta is per-ply discount.
    Uses lru_cache over (to_move, canon_board, invX, invO, h, p_ref).
    """
    @lru_cache(maxsize=200000)
    def rec(to_move: int, canon_board: Tuple[int,...], invX: Tuple[int,int,int], invO: Tuple[int,int,int], h: int, p_ref: int) -> float:
        st = Z.State(canon_board, invX, invO, to_move)
        term, w = Z.is_terminal(st.board)
        if term:
            return 1.0 if w==p_ref else -1.0 if w==Z.other(p_ref) else 0.0
        if h == 0:
            return 0.0
        key_ccp = (to_move, (canon_board, invX, invO))
        probs = ccp.get(key_ccp, None)
        legal = Z.legal_moves(st)
        if not legal:
            return 0.0
        ev = 0.0
        if probs is None:
            pa = 1.0/len(legal)
            for mv in legal:
                s2 = Z.apply_move(st, mv)
                bcan = canonical_key_board(s2)
                ev += pa * rec(s2.to_move, bcan, s2.invX, s2.invO, h-1, p_ref)
        else:
            for mv in legal:
                aid = action_id(mv[0], mv[1])
                pa = probs[aid]
                if pa <= 0: continue
                s2 = Z.apply_move(st, mv)
                bcan = canonical_key_board(s2)
                ev += pa * rec(s2.to_move, bcan, s2.invX, s2.invO, h-1, p_ref)
        return beta * ev

    b0 = canonical_key_board(state)
    return rec(state.to_move, b0, state.invX, state.invO, H, p_reference)

def build_offsets(dataset: List[Datum], ccp: Dict, H: int=6, beta: float=1.0) -> List[np.ndarray]:
    """
    Returns list of 27-dim arrays C where C[a] = EV_from_state(T(s,a), p_ref, ccp, H, beta)
    for legal a, and a large negative number for illegal a (for masking).
    """
    offsets: List[np.ndarray] = []
    for d in dataset:
        s = d.state; p_ref = s.to_move
        C = np.full(27, -1e9, dtype=float)
        legal = Z.legal_moves(s)
        if not legal:
            offsets.append(C); continue
        for mv in legal:
            aid = action_id(mv[0], mv[1])
            s2 = Z.apply_move(s, mv)
            C[aid] = EV_from_state(s2, p_ref, ccp, H=H, beta=beta)
        offsets.append(C)
    return offsets

# -------- Features: next-state features --------
def features_of_next(s: Z.State, a_id: int) -> Optional[np.ndarray]:
    v,i = id_to_action(a_id)
    if (v,i) not in Z.legal_moves(s):
        return None
    s2 = Z.apply_move(s, (v,i))
    return Z.feature_vector(s2)

# -------- Logit with offsets --------
def _softmax_with_mask(logits: np.ndarray, mask: np.ndarray):
    x = np.where(mask, logits, -1e9)
    m = np.max(x)
    ex = np.exp(x - m)
    ex = np.where(mask, ex, 0.0)
    Zs = ex.sum()
    if Zs <= 0:
        p = mask.astype(float); p /= max(1.0, p.sum())
        return p
    return ex / Zs

def fit_theta(dataset: List[Datum], offsets: List[np.ndarray], beta: float=1.0, lr: float=0.2, iters: int=600):
    # infer K
    K = None
    for d in dataset:
        f = features_of_next(d.state, d.a_id)
        if f is not None:
            K = len(f); break
    if K is None:
        raise ValueError("Cannot infer feature length; dataset seems empty or illegal.")
    theta = np.zeros(K, dtype=float)
    for it in range(iters):
        g = np.zeros_like(theta); ll = 0.0
        for d, C in zip(dataset, offsets):
            mask = legal_mask_27(d.state)
            logits = np.full(27, -1e9, dtype=float)
            feats  = [None]*27
            for a in range(27):
                if not mask[a]: continue
                φ = features_of_next(d.state, a)
                feats[a] = φ
                logits[a] = float(np.dot(theta, φ)) + beta*C[a]
            p = _softmax_with_mask(logits, mask)
            ll += np.log(p[d.a_id] + 1e-12)
            for a in range(27):
                if not mask[a]: continue
                g += ((1.0 if a==d.a_id else 0.0) - p[a]) * feats[a]
        theta += lr * g / max(1, len(dataset))
        if (it+1) % 100 == 0:
            print(f"iter {it+1:4d}  avg ll={ll/len(dataset):.4f}")
    return theta

def loglik(dataset: List[Datum], offsets: List[np.ndarray], theta: np.ndarray, beta: float=1.0) -> float:
    ll = 0.0
    for d, C in zip(dataset, offsets):
        mask = legal_mask_27(d.state)
        logits = np.full(27, -1e9, dtype=float)
        for a in range(27):
            if not mask[a]: continue
            φ = features_of_next(d.state, a)
            logits[a] = float(np.dot(theta, φ)) + beta*C[a]
        p = _softmax_with_mask(logits, mask)
        ll += np.log(p[d.a_id] + 1e-12)
    return ll

def policy_probs(s: Z.State, theta: np.ndarray, ccp: Dict, H: int=6, beta: float=1.0):
    C = np.full(27, -1e9, dtype=float)
    legal = Z.legal_moves(s)
    for mv in legal:
        aid = action_id(mv[0], mv[1])
        C[aid] = EV_from_state(Z.apply_move(s, mv), s.to_move, ccp, H=H, beta=beta)
    mask = legal_mask_27(s)
    logits = np.full(27, -1e9, dtype=float)
    for a in range(27):
        if not mask[a]: continue
        φ = features_of_next(s, a)
        logits[a] = float(np.dot(theta, φ)) + beta*C[a]
    return _softmax_with_mask(logits, mask)

def train_test_split(data: List[Datum], test_ratio: float=0.2, seed: int=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    cut = int(len(idx) * (1.0 - test_ratio))
    tr = [data[i] for i in idx[:cut]]
    te = [data[i] for i in idx[cut:]]
    return tr, te
