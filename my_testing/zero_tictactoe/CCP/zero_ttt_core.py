
from dataclasses import dataclass
import numpy as np

EMPTY = 0
X1, X2, X3 = 1, 2, 3
O1, O2, O3 = 4, 5, 6
X, O = 1, 2
OWNER = [0, X, X, X, O, O, O]
VAL   = [0, 1, 2, 3, 1, 2, 3]
INIT_INV = (2,2,2)
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

@dataclass(frozen=True)
class State:
    board: tuple
    invX: tuple
    invO: tuple
    to_move: int

def other(p): return X if p==O else O
def encode(p,v): return [None, X1,X2,X3][v] if p==X else [None, O1,O2,O3][v]

def is_terminal(board):
    for a,b,c in WIN_LINES:
        if board[a]!=EMPTY and OWNER[board[a]]==OWNER[board[b]]==OWNER[board[c]]:
            return True, OWNER[board[a]]
    return False, 0

def legal_moves(state: State):
    board, p = state.board, state.to_move
    inv = state.invX if p==X else state.invO
    out = []
    for i,cell in enumerate(board):
        own = OWNER[cell]; val = VAL[cell]
        for v in (1,2,3):
            if inv[v-1] <= 0: continue
            if cell==EMPTY: out.append((v,i))
            elif own!=p and v>val: out.append((v,i))
    return out

def apply_move(state: State, mv):
    v,i = mv; p = state.to_move
    b = list(state.board); b[i] = encode(p,v)
    if p==X:
        invX = (state.invX[0]-(v==1), state.invX[1]-(v==2), state.invX[2]-(v==3))
        invO = state.invO
    else:
        invO = (state.invO[0]-(v==1), state.invO[1]-(v==2), state.invO[2]-(v==3))
        invX = state.invX
    return State(tuple(b), invX, invO, other(p))

def initial_state(): return State(tuple([EMPTY]*9), INIT_INV, INIT_INV, X)

# Features
FEATURE_NAMES = ["center","corners","sides","two_open","neg_opp_two_open",
                 "inv3_diff","inv2_diff","inv1_diff","ow_lt1","ow_lt2","ow_lt3","secure_cells",
                 "win_now","need_block"]

def two_in_row_open(board, player, inv_self, inv_opp):
    cnt=0
    for a,b,c in WIN_LINES:
        owners=[OWNER[board[a]], OWNER[board[b]], OWNER[board[c]]]
        if owners.count(player)==2 and owners.count(0)==1:
            if sum(inv_self)>0: cnt+=1
    return cnt

def overwritable_counts(board, player):
    opp = other(player); lt1=lt2=lt3=0
    for cell in board:
        if cell==EMPTY: continue
        if OWNER[cell]==opp:
            if VAL[cell] < 1: lt1+=1
            if VAL[cell] < 2: lt2+=1
            if VAL[cell] < 3: lt3+=1
    return (lt1,lt2,lt3)

def secure_cells(board, player, inv_opp):
    max_opp = 3 if inv_opp[2]>0 else 2 if inv_opp[1]>0 else 1 if inv_opp[0]>0 else 0
    return sum(1 for c in board if c!=EMPTY and OWNER[c]==player and VAL[c]>=max_opp)

def feature_vector(state: State):
    board, p = state.board, state.to_move
    inv_self = state.invX if p==X else state.invO
    inv_opp  = state.invO if p==X else state.invX
    center = 1 if (board[4]!=EMPTY and OWNER[board[4]]==p) else 0
    corners = sum(1 for i in (0,2,6,8) if (board[i]!=EMPTY and OWNER[board[i]]==p))
    sides = sum(1 for i in (1,3,5,7) if (board[i]!=EMPTY and OWNER[board[i]]==p))
    two_open = two_in_row_open(board, p, inv_self, inv_opp)
    opp_two_open = two_in_row_open(board, other(p), inv_opp, inv_self)
    inv_diff = (inv_self[2]-inv_opp[2], inv_self[1]-inv_opp[1], inv_self[0]-inv_opp[0])
    lt1,lt2,lt3 = overwritable_counts(board, p)
    secure = secure_cells(board, p, inv_opp)
    win_now = can_win_now(state)
    need_block = opponent_has_immediate_win(state)
    return np.array([center, corners, sides, two_open, -opp_two_open,
                     inv_diff[0], inv_diff[1], inv_diff[2], lt1, lt2, lt3, secure,
                     win_now, need_block], dtype=float)

def E_theta(state: State, theta):
    x = feature_vector(state); th = np.asarray(theta, dtype=float)
    k = min(len(x), len(th)); return float(np.dot(x[:k], th[:k]))

def children_ordered_by_theta(state: State, theta):
    moves = legal_moves(state)
    def key(mv):
        v,i = mv; p = state.to_move
        cur = state.board[i]
        win_now = 0
        if cur==0 or (OWNER[cur]!=p and VAL[cur]<v):
            b=list(state.board); b[i] = encode(p,v)
            term,w = is_terminal(tuple(b))
            if term and w==p: win_now=1
        s2 = apply_move(state, mv)
        return (win_now, E_theta(s2, theta))
    return sorted(moves, key=key, reverse=True)

def search_theta(state: State, depth, theta, alpha=-2, beta=2):
    term, w = is_terminal(state.board)
    if term: return (+1 if w==state.to_move else -1 if w==other(state.to_move) else 0), None
    if depth==0 or (sum(state.invX)==0 and sum(state.invO)==0):
        return E_theta(state, theta), None
    best=-2; best_mv=None
    for mv in children_ordered_by_theta(state, theta):
        s2 = apply_move(state, mv)
        val,_ = search_theta(s2, depth-1, theta, -beta, -alpha)
        val = -val
        if val>best: best=val; best_mv=mv
        alpha = max(alpha, val)
        if alpha>=beta: break
    return best, best_mv


def can_win_now(state: State) -> int:
    p = state.to_move
    for (v,i) in legal_moves(state):
        b = list(state.board); cur = b[i]
        if cur==EMPTY or (OWNER[cur]!=p and VAL[cur]<v):
            b[i] = encode(p,v)
            term, w = is_terminal(tuple(b))
            if term and w==p: return 1
    return 0

def opponent_has_immediate_win(state: State) -> int:
    # after opponent move from current state, can they win immediately?
    opp = other(state.to_move)
    # simulate opponent's turn without changing inventories: we must use true apply_move to reduce stock
    for (v,i) in legal_moves(state):
        s2 = apply_move(state, (v,i))  # now opp to move
        for (vv,j) in legal_moves(s2):
            b = list(s2.board); cur = b[j]
            if cur==EMPTY or (OWNER[cur]!=s2.to_move and VAL[cur]<vv):
                b[j] = encode(s2.to_move, vv)
                term, w = is_terminal(tuple(b))
                if term and w==s2.to_move: return 1
    return 0
