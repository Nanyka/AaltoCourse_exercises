
from dataclasses import dataclass
import numpy as np

# cell values: 0 empty, 1:X1, 2:X2, 3:X3, 4:O1, 5:O2, 6:O3
EMPTY = 0
X1, X2, X3 = 1, 2, 3
O1, O2, O3 = 4, 5, 6
X, O = 1, 2
OWNER = {0:0, 1:X,2:X,3:X, 4:O,5:O,6:O}
VAL   = {0:0, 1:1, 2:2, 3:3, 4:1, 5:2, 6:3}
INIT_INV = (2,2,2)
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

# symmetries
ROT90 = (6,3,0,7,4,1,8,5,2)
ROT180 = tuple(ROT90[i] for i in ROT90)
ROT270 = tuple(ROT180[i] for i in ROT90)
REF_H = (2,1,0,5,4,3,8,7,6)
REF_V = (6,7,8,3,4,5,0,1,2)
REF_D = (0,3,6,1,4,7,2,5,8)
REF_AD= (8,5,2,7,4,1,6,3,0)
SYMS = [tuple(range(9)), ROT90, ROT180, ROT270, REF_H, REF_V, REF_D, REF_AD]

@dataclass(frozen=True)
class State:
    board: tuple
    invX: tuple
    invO: tuple
    to_move: int
    def pieces_used(self):
        return (6 - sum(self.invX)) + (6 - sum(self.invO))

def other(p): return X if p==O else O
def encode(p,v): return [None, X1,X2,X3][v] if p==X else [None, O1,O2,O3][v]

def is_terminal(board):
    for a,b,c in WIN_LINES:
        if board[a]!=EMPTY and OWNER[board[a]]==OWNER[board[b]]==OWNER[board[c]]:
            return True, OWNER[board[a]]
    return False, 0

def win_if(board, p, i, v):
    b = list(board)
    cur = board[i]
    if cur!=EMPTY:
        if OWNER[cur]==p or VAL[cur] >= v: 
            return False
    b[i] = encode(p, v)
    for a,b1,c in WIN_LINES:
        if b[a]!=EMPTY and OWNER[b[a]]==OWNER[b[b1]]==OWNER[b[c]]:
            return True
    return False

def legal_moves(state: State):
    board, invX, invO, p = state.board, state.invX, state.invO, state.to_move
    inv = invX if p==X else invO
    moves = []
    for i in range(9):
        cell = board[i]
        owner = OWNER[cell]
        val   = VAL[cell]
        for v in (1,2,3):
            if inv[v-1] <= 0: continue
            if cell==EMPTY:
                moves.append((v,i))
            else:
                if owner==p: continue
                if v>val: moves.append((v,i))
    return moves

def apply_move(state: State, move):
    v,i = move
    p = state.to_move
    board = list(state.board)
    board[i] = encode(p, v)
    if p==X:
        invX = (state.invX[0]-(1 if v==1 else 0),
                state.invX[1]-(1 if v==2 else 0),
                state.invX[2]-(1 if v==3 else 0))
        invO = state.invO
    else:
        invO = (state.invO[0]-(1 if v==1 else 0),
                state.invO[1]-(1 if v==2 else 0),
                state.invO[2]-(1 if v==3 else 0))
        invX = state.invX
    return State(tuple(board), invX, invO, other(p))

def transform_board(board, perm):
    return tuple(board[perm[k]] for k in range(9))

def canonical_state(state: State):
    boards = [ transform_board(state.board, perm) for perm in SYMS ]
    best_board = min(boards)
    return (best_board, state.invX, state.invO, state.to_move)

CENTER = 4
CORNERS = {0,2,6,8}
SIDES = {1,3,5,7}

def order_key(state: State, move):
    v,i = move
    p = state.to_move
    win_now  = 1 if win_if(state.board, p, i, v) else 0
    shape = 2 if i==CENTER else (1 if i in CORNERS else 0)
    return (win_now, shape, v)

def solve_exact(state: State):
    tt = {}
    def negamax(s, alpha, beta):
        key = canonical_state(s)
        term, w = is_terminal(s.board)
        if term:
            return (+1 if w==s.to_move else -1 if w==other(s.to_move) else 0), None
        if sum(s.invX)==0 and sum(s.invO)==0:
            return 0, None
        if key in tt: return tt[key]
        moves = legal_moves(s)
        if not moves: return 0, None
        moves.sort(key=lambda a: order_key(s,a), reverse=True)
        best = -2; best_move = moves[0]
        for mv in moves:
            s2 = apply_move(s, mv)
            val,_ = negamax(s2, -beta, -alpha)
            val = -val
            if val>best: best=val; best_move=mv
            alpha = max(alpha, val)
            if alpha>=beta: break
        tt[key]=(best, best_move)
        return best, best_move
    return negamax(state, -2, 2)

def two_in_row_open(board, player, inv_self, inv_opp):
    cnt=0
    for a,b,c in WIN_LINES:
        cells=[a,b,c]
        owners=[OWNER[board[k]] for k in cells]
        if owners.count(player)==2 and owners.count(0)==1:
            if sum(inv_self)>0: cnt+=1
    return cnt

def overwritable_counts(board, player):
    opp = X if player==O else O
    less = [0,0,0]
    for cell in board:
        if cell==0: continue
        if OWNER[cell]==opp:
            if VAL[cell]<1: less[0]+=1
            if VAL[cell]<2: less[1]+=1
            if VAL[cell]<3: less[2]+=1
    return np.array(less,dtype=float)

def secure_cells(board, player, inv_opp):
    max_opp = (3 if inv_opp[2]>0 else 2 if inv_opp[1]>0 else 1 if inv_opp[0]>0 else 0)
    cnt=0
    for cell in board:
        if cell==0: continue
        if OWNER[cell]==player and VAL[cell]>=max_opp: cnt+=1
    return cnt

def feature_vector(state):
    board, invX, invO, p = state.board, state.invX, state.invO, state.to_move
    inv_self = invX if p==X else invO
    inv_opp  = invO if p==X else invX
    center = 1 if OWNER[board[4]]==p else 0
    corners = sum(1 for i in (0,2,6,8) if OWNER[board[i]]==p)
    sides = sum(1 for i in (1,3,5,7) if OWNER[board[i]]==p)
    two_open = two_in_row_open(board, p, inv_self, inv_opp)
    opp_two_open = two_in_row_open(board, other(p), inv_opp, inv_self)
    inv_feat = np.array(inv_self) - np.array(inv_opp)
    owc = overwritable_counts(board, p)
    secure = secure_cells(board, p, inv_opp)
    return np.array([center, corners, sides, two_open, -opp_two_open,
                     inv_feat[2], inv_feat[1], inv_feat[0],
                     owc[0], owc[1], owc[2], secure], dtype=float)

FEATURE_NAMES = ["center","corners","sides","two_open","neg_opp_two_open",
                 "inv3_diff","inv2_diff","inv1_diff","ow_lt1","ow_lt2","ow_lt3","secure_cells"]

def E_theta(state, theta): 
    return float(np.dot(theta, feature_vector(state)))

def children_ordered_by_theta(state, theta):
    moves = legal_moves(state)
    def score(mv):
        v,i = mv
        p = state.to_move
        win_now = 1 if win_if(state.board, p, i, v) else 0
        s2 = apply_move(state, mv)
        return (win_now, E_theta(s2, theta))
    moves.sort(key=score, reverse=True)
    return moves

def search_theta(state, depth, theta, alpha=-2, beta=2, tt=None):
    if tt is None: tt={}
    key=(canonical_state(state), depth)
    term,w = is_terminal(state.board)
    if term: return (+1 if w==state.to_move else -1 if w==other(state.to_move) else 0), None
    if depth==0 or (sum(state.invX)==0 and sum(state.invO)==0):
        return E_theta(state, theta), None
    if key in tt: return tt[key]
    best=-2; best_move=None
    for mv in children_ordered_by_theta(state, theta):
        s2 = apply_move(state, mv)
        val,_ = search_theta(s2, depth-1, theta, -beta, -alpha, tt)
        val = -val
        if val>best: best=val; best_move=mv
        alpha=max(alpha,val)
        if alpha>=beta: break
    tt[key]=(best,best_move)
    return best,best_move

def initial_state():
    return State(tuple([EMPTY]*9), INIT_INV, INIT_INV, X)

def random_reachable_state(seed=0, steps=0):
    rng = np.random.default_rng(seed)
    s = initial_state()
    if steps<=0: steps=int(rng.integers(0,12))
    for t in range(steps):
        mv_list = legal_moves(s)
        if not mv_list: break
        mv = mv_list[int(rng.integers(0,len(mv_list)))]
        s = apply_move(s, mv)
        term,_ = is_terminal(s.board)
        if term: break
    return s
