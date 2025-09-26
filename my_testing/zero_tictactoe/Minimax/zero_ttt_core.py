
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Iterable, List

# =========================
#   Zero Tic‑Tac‑Toe core
# =========================

# ----- Encodings -----
EMPTY = 0
X1, X2, X3 = 1, 2, 3
O1, O2, O3 = 4, 5, 6
X, O = 1, 2

# OWNER[c]: {0=empty,1=X,2=O} ; VAL[c]: piece magnitude
OWNER = [0, X, X, X, O, O, O]
VAL   = [0, 1, 2, 3, 1, 2, 3]

# Initial stock: two of each value
INIT_INV = (2, 2, 2)

# Lines that win by owner (values may differ)
WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

#   Feature set (14)
FEATURE_NAMES = [
    "center", "corners", "sides",
    "two_open", "neg_opp_two_open",
    "inv3_diff", "inv2_diff", "inv1_diff",
    "ow_lt1", "ow_lt2", "ow_lt3",
    "secure_cells",
    "win_now", "need_block"
]

@dataclass(frozen=True)
class State:
    board: tuple     # length-9 ints in 0..6
    invX:  tuple     # (c1,c2,c3)
    invO:  tuple     # (c1,c2,c3)
    to_move: int     # X or O

def other(p: int) -> int:
    return X if p == O else O

def encode(p: int, v: int) -> int:
    return [None, X1, X2, X3][v] if p == X else [None, O1, O2, O3][v]

# ----------------
#  Game mechanics
# ----------------
def is_terminal(board: Tuple[int,...]):
    """Return (terminal_bool, winner_owner). Winner_owner in {0,1,2}, where 0 means draw/not terminal."""
    for a,b,c in WIN_LINES:
        if board[a] != EMPTY and OWNER[board[a]] == OWNER[board[b]] == OWNER[board[c]]:
            return True, OWNER[board[a]]
    return False, 0

def legal_moves(state: State):
    """All legal (v, i) for the side to move.
       - Can place on empty.
       - Can overwrite opponent if v > opponent value.
       - Cannot overwrite own piece.
       - Must have inventory for chosen v.
    """
    board, p = state.board, state.to_move
    inv = state.invX if p == X else state.invO
    out = []
    for i, cell in enumerate(board):
        if cell == EMPTY:
            if inv[0] > 0: out.append((1, i))
            if inv[1] > 0: out.append((2, i))
            if inv[2] > 0: out.append((3, i))
        else:
            own = OWNER[cell]; val = VAL[cell]
            if own != p:
                if inv[1] > 0 and 2 > val: out.append((2, i))
                if inv[2] > 0 and 3 > val: out.append((3, i))
    return out

def apply_move(state: State, mv: Tuple[int,int]) -> State:
    v, i = mv; p = state.to_move
    b = list(state.board); b[i] = encode(p, v)
    if p == X:
        invX = (state.invX[0] - (v==1), state.invX[1] - (v==2), state.invX[2] - (v==3))
        invO = state.invO
    else:
        invO = (state.invO[0] - (v==1), state.invO[1] - (v==2), state.invO[2] - (v==3))
        invX = state.invX
    return State(tuple(b), invX, invO, other(p))

def initial_state() -> State:
    return State(tuple([EMPTY]*9), INIT_INV, INIT_INV, X)


def _two_in_row_open(board: Tuple[int,...], player: int, inv_self: Tuple[int,int,int]) -> int:
    cnt = 0
    if sum(inv_self) <= 0:
        return 0
    for a,b,c in WIN_LINES:
        owners = [OWNER[board[a]], OWNER[board[b]], OWNER[board[c]]]
        if owners.count(player) == 2 and owners.count(0) == 1:
            cnt += 1
    return cnt

def _overwritable_counts(board: Tuple[int,...], player: int):
    opp = other(player); lt1 = 0; lt2 = 0; lt3 = 0
    for cell in board:
        if cell == EMPTY or OWNER[cell] != opp: 
            continue
        v = VAL[cell]
        if v < 2: lt2 += 1     # opponent 1s
        if v < 3: lt3 += 1     # opponent 1s or 2s
    return (lt1, lt2, lt3)

def _secure_cells(board: Tuple[int,...], player: int, inv_opp: Tuple[int,int,int]) -> int:
    max_opp = 3 if inv_opp[2] > 0 else 2 if inv_opp[1] > 0 else 1 if inv_opp[0] > 0 else 0
    return sum(1 for c in board if c != EMPTY and OWNER[c] == player and VAL[c] >= max_opp)

def _can_win_now(state: State) -> int:
    p = state.to_move
    for (v,i) in legal_moves(state):
        b = list(state.board)
        cur = b[i]
        if cur == EMPTY or (OWNER[cur] != p and VAL[cur] < v):
            b[i] = encode(p, v)
            term, w = is_terminal(tuple(b))
            if term and w == p:
                return 1
    return 0

def _opponent_can_win_now(state: State) -> int:
    opp = other(state.to_move)
    fake_state = State(state.board, state.invX, state.invO, opp)
    for (v,i) in legal_moves(fake_state):
        b = list(state.board)
        cur = b[i]
        if cur == EMPTY or (OWNER[cur] != opp and VAL[cur] < v):
            b[i] = encode(opp, v)
            term, w = is_terminal(tuple(b))
            if term and w == opp:
                return 1
    return 0

def feature_vector(state: State) -> np.ndarray:
    board, p = state.board, state.to_move
    inv_self = state.invX if p == X else state.invO
    inv_opp  = state.invO if p == X else state.invX

    center  = 1 if (board[4] != EMPTY and OWNER[board[4]] == p) else 0
    corners = sum(1 for i in (0,2,6,8) if board[i] != EMPTY and OWNER[board[i]] == p)
    sides   = sum(1 for i in (1,3,5,7) if board[i] != EMPTY and OWNER[board[i]] == p)

    two_open      = _two_in_row_open(board, p, inv_self)
    opp_two_open  = _two_in_row_open(board, other(p), inv_opp)
    inv3_diff, inv2_diff, inv1_diff = (inv_self[2]-inv_opp[2], inv_self[1]-inv_opp[1], inv_self[0]-inv_opp[0])
    lt1, lt2, lt3 = _overwritable_counts(board, p)
    secure        = _secure_cells(board, p, inv_opp)
    win_now       = _can_win_now(state)
    need_block    = _opponent_can_win_now(state)

    x = np.array([
        center, corners, sides,
        two_open, -opp_two_open,
        inv3_diff, inv2_diff, inv1_diff,
        lt1, lt2, lt3,
        secure,
        win_now, need_block
    ], dtype=float)
    assert len(x) == 14, "Feature vector must be length 14."
    return x

def E_theta(state: State, theta) -> float:
    x = feature_vector(state)
    th = np.asarray(theta, dtype=float)
    if len(th) != len(x):
        raise ValueError(f"theta length {len(th)} != feature length {len(x)} (expected 14).")
    return float(np.dot(x, th))

def children_ordered_by_theta(state: State, theta):
    moves = legal_moves(state)
    p = state.to_move
    def key(mv):
        v, i = mv
        b = list(state.board); cur = b[i]
        if cur == EMPTY or (OWNER[cur] != p and VAL[cur] < v):
            b[i] = encode(p, v)
            term, w = is_terminal(tuple(b))
            if term and w == p:
                return (1, 0.0)
        s2 = apply_move(state, mv)
        return (0, E_theta(s2, theta))
    return sorted(moves, key=key, reverse=True)

def search_theta(state: State, depth: int, theta, alpha: float=-2.0, beta: float=2.0):
    term, w = is_terminal(state.board)
    if term:
        return (+1 if w == state.to_move else -1 if w == other(state.to_move) else 0.0), None
    if depth == 0 or (sum(state.invX) == 0 and sum(state.invO) == 0):
        return E_theta(state, theta), None

    best = -2.0; best_mv = None
    for mv in children_ordered_by_theta(state, theta):
        s2 = apply_move(state, mv)
        val, _ = search_theta(s2, depth-1, theta, -beta, -alpha)
        val = -val
        if val > best:
            best, best_mv = val, mv
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best, best_mv
