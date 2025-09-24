
import numpy as np

# ---------- Constants & basic utils ----------
EMPTY, X, O = 0, 1, 2
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]
CORNERS, SIDES = {0,2,6,8}, {1,3,5,7}

def legal_moves(board): return [i for i,v in enumerate(board) if v==EMPTY]

def apply_move(board, move, player):
    b = list(board); b[move] = player; return tuple(b)

def other(player): return X if player==O else O

def check_winner(board):
    for a,b,c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a], False
    if EMPTY not in board: return 0, True
    return 0, False

# ---------- Rendering ----------
def board_to_str(board):
    """Return pretty 3x3 string for a 9-length state tuple/list."""
    syms = {EMPTY:'.', X:'X', O:'O'}
    s = ''.join(syms[v] for v in board)
    return '\n'.join([' '.join(s[0:3]), ' '.join(s[3:6]), ' '.join(s[6:9])])

def print_trajectory(trajectory):
    """trajectory: list of (board, player, action)."""
    for t,(b,p,a) in enumerate(trajectory, start=1):
        who = 'X' if p==X else 'O'
        print(f"Move {t}: Player {who} -> {a}")
        print(board_to_str(b)); print()

def index_map():
    return "0 1 2\n3 4 5\n6 7 8"

# ---------- Features for structural model ----------
def wins_if(board, move, player):
    b = apply_move(board, move, player)
    w,_ = check_winner(b)
    return w == player

def blocks_if(board, move, player):
    opp = other(player)
    for m in legal_moves(board):
        if wins_if(board, m, opp):
            return move == m
    return False

def create_two(board, move, player):
    b = apply_move(board, move, player)
    c = 0
    for a,b1,d in WIN_LINES:
        line = [b[a], b[b1], b[d]]
        if line.count(player)==2 and line.count(EMPTY)==1:
            c += 1
    return 1 if c>=1 else 0

FEATURE_NAMES = ["win_now","block_now","center","corner","side","create_two"]

def features(board, move, player):
    return np.array([
        1 if wins_if(board, move, player) else 0,
        1 if blocks_if(board, move, player) else 0,
        1 if move==4 else 0,
        1 if move in CORNERS else 0,
        1 if move in SIDES else 0,
        create_two(board, move, player)
    ], dtype=float)

# ---------- Policies ----------
def safe_softmax(z):
    z = np.asarray(z, dtype=float)
    if z.size==0: return z
    z -= np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    if not np.isfinite(s) or s <= 0:  # numeric guard
        return np.ones_like(z) / len(z)
    return ez / s

def logits_for_state(board, player, theta):
    moves = legal_moves(board)
    X = np.stack([features(board, m, player) for m in moves], axis=0) if moves else np.zeros((0, len(theta)))
    z = X @ theta if moves else np.array([])
    return z, moves, X

def sample_policy_move(board, player, theta, rng):
    z, moves, X = logits_for_state(board, player, theta)
    if not moves: return None
    p = safe_softmax(z)
    try:
        return int(rng.choice(moves, p=p))
    except Exception:
        return int(rng.choice(moves))

def greedy_policy_move(board, player, theta):
    z, moves, X = logits_for_state(board, player, theta)
    if not moves: return None
    return int(moves[int(np.argmax(z))])

# ---------- Play loops ----------
def play_game(theta, first_player=X, seed=None, max_plies=9, stochastic=True):
    """Return (trajectory, winner). If stochastic=False, bot plays greedily."""
    rng = np.random.default_rng(seed)
    board = tuple([EMPTY]*9)
    player = first_player
    trajectory = []
    plies = 0
    while True:
        w, draw = check_winner(board)
        if w!=0 or draw: return trajectory, w
        if plies >= max_plies: return trajectory, 0  # safety
        mv = sample_policy_move(board, player, theta, rng) if stochastic else greedy_policy_move(board, player, theta)
        if mv is None: return trajectory, 0
        trajectory.append((board, player, mv))
        board = apply_move(board, mv, player)
        player = other(player)
        plies += 1

def human_vs_bot(theta, bot_player='O', stochastic=False, input_fn=input):
    """Play in the terminal/notebook. Greedy bot by default; set stochastic=True for sampling."""
    bot = X if bot_player.upper()=='X' else O
    human = O if bot==X else X

    print("Index map:\n" + index_map() + "\n")
    board = tuple([EMPTY]*9)
    player = X
    print("Start:\n" + board_to_str(board) + "\n")

    while True:
        w, draw = check_winner(board)
        if w!=0 or draw:
            if w==bot: print("Bot wins!")
            elif w==human: print("You win!")
            else: print("Draw!")
            print("\nFinal board:\n" + board_to_str(board))
            return w

        if player == human:
            lm = legal_moves(board)
            while True:
                try:
                    mv = int(input_fn(f"Your move ({'X' if human==X else 'O'}), choose {lm}: "))
                    if mv in lm: break
                    print("Illegal move.")
                except Exception:
                    print("Enter an integer index.")
            board = apply_move(board, mv, human)
        else:
            mv = (sample_policy_move(board, bot, theta, np.random.default_rng())
                  if stochastic else greedy_policy_move(board, bot, theta))
            board = apply_move(board, mv, bot)
            print(f"Bot plays {mv}:\n{board_to_str(board)}\n")
        player = other(player)
