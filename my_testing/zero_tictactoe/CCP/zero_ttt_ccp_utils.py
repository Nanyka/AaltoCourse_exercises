
# zero_ttt_ccp_utils.py
# Helpers to play Zero TTT using a CCP policy + linear features θ.
# Requires: zero_ttt_core.py, ccp_zero_ttt.py

from typing import Optional, Tuple, List, Dict
import numpy as np
import zero_ttt_core as Z
import ccp_zero_ttt as CCP

# ------------ pretty printing ------------
SYMBOL = {0:"·", 1:"X1", 2:"X2", 3:"X3", 4:"O1", 5:"O2", 6:"O3"}

def print_board(board: Tuple[int,...]):
    rows = []
    for r in range(3):
        i = 3*r
        row = " | ".join(f"{SYMBOL[board[i+c]]:^3}" for c in range(3))
        rows.append(row)
    sep = "\n" + "-"*15 + "\n"
    print(rows[0] + sep + rows[1] + sep + rows[2])

def print_state(s: Z.State):
    print_board(s.board)
    x_inv = f"X inv: 1×{s.invX[0]}  2×{s.invX[1]}  3×{s.invX[2]}"
    o_inv = f"O inv: 1×{s.invO[0]}  2×{s.invO[1]}  3×{s.invO[2]}"
    to = "X" if s.to_move==Z.X else "O"
    print(x_inv + "     " + o_inv + f"     to_move: {to}")

# ------------ policy selection ------------
def choose_move_ccp(s: Z.State, theta: np.ndarray, ccp: Dict, H: int=3, beta: float=1.0,
                    mode: str = "argmax", rng: Optional[np.random.Generator]=None,
                    lookahead_depth: int = 2, lambda_lookahead: float = 0.5) -> Optional[Tuple[int,int]]:
    """
    Pick a move using CCP policy with offsets and linear θ features.
    mode: "argmax" (default) or "sample"
    """
    probs = CCP.policy_probs(s, theta=theta, ccp=ccp, H=H, beta=beta)
    # optional: blend with a fast lookahead on each legal move
    def lookahead_bonus(mv):
        if lookahead_depth<=0: return 0.0
        s2 = Z.apply_move(s, mv)
        val,_ = Z.search_theta(s2, depth=lookahead_depth, theta=theta)
        return float(val)
    # mask illegal (should already be masked), then pick
    legal = Z.legal_moves(s)
    if not legal:
        return None
    # Map probs -> list over these legal moves
    legal_ids = [CCP.action_id(v,i) for (v,i) in legal]
    base = np.array([probs[a] for a in legal_ids], dtype=float)
    if lookahead_depth>0:
        la = np.array([lookahead_bonus(mv) for mv in legal], dtype=float)
        # shift/scale lookahead to [0,1] before blending
        if np.isfinite(la).any():
            la = (la - la.min()) / (la.max()-la.min() + 1e-9)
        else:
            la = np.zeros_like(base)
        ps = (1.0 - lambda_lookahead) * base + lambda_lookahead * la
    else:
        ps = base
    ssum = ps.sum()
    if ssum <= 0:
        # fall back to uniform over legal
        ps = np.ones(len(legal), dtype=float) / len(legal)
    else:
        ps = ps / ssum

    if mode == "sample":
        rng = rng or np.random.default_rng()
        ix = int(rng.choice(len(legal), p=ps))
        return legal[ix]
    # argmax
    ix = int(np.argmax(ps))
    return legal[ix]

# ------------ gameplay ------------
def play_game(theta_X: np.ndarray, theta_O: Optional[np.ndarray], ccp: Dict,
              H: int=3, beta: float=1.0,
              mode_X: str="argmax", mode_O: str="argmax",
              seed: Optional[int]=None, verbose: bool=False):
    """
    Bot vs Bot using CCP policies. If theta_O is None, use theta_X for both sides.
    Returns: winner (1=X, 2=O, 0=draw), history (list of (state_before, move)), final_state
    """
    rng = np.random.default_rng(seed)
    if theta_O is None:
        theta_O = theta_X
    s = Z.initial_state()
    history = []
    while True:
        term, w = Z.is_terminal(s.board)
        # draw if no inventories and no win OR no legal move for current player
        no_moves = (len(Z.legal_moves(s))==0)
        if term or no_moves or (sum(s.invX)==0 and sum(s.invO)==0):
            return (w if term else 0), history, s
        # act
        theta = theta_X if s.to_move==Z.X else theta_O
        mode = mode_X if s.to_move==Z.X else mode_O
        mv = choose_move_ccp(s, theta=theta, ccp=ccp, H=H, beta=beta, mode=mode, rng=rng)
        if mv is None:
            return 0, history, s  # draw (no legal move)
        history.append((s, mv))
        s = Z.apply_move(s, mv)
        if verbose:
            print_state(s)

def human_vs_bot(ccp: Dict, theta_bot: np.ndarray, human: str="X",
                 H: int=3, beta: float=1.0, mode_bot: str="argmax"):
    """
    Simple CLI loop to play a game against the CCP bot.
    Input format: v i  (value in {1,2,3}, index i in {0..8})
    """
    human = human.upper()
    assert human in ("X","O")
    human_p = Z.X if human=="X" else Z.O
    s = Z.initial_state()
    print("Welcome to Zero TTT (CCP bot). You are", human)
    print("Cells are indexed 0..8:")
    print("0 1 2\n3 4 5\n6 7 8")
    print_state(s)

    while True:
        term, w = Z.is_terminal(s.board)
        no_moves = (len(Z.legal_moves(s))==0)
        if term or no_moves or (sum(s.invX)==0 and sum(s.invO)==0):
            if term:
                print("Game over:", "X wins" if w==Z.X else "O wins")
                return w
            else:
                print("Game over: draw")
                return 0

        if s.to_move == human_p:
            # human turn
            legal = set(Z.legal_moves(s))
            try:
                raw = input("Your move (v i): ").strip()
                if raw.lower() in ("q","quit","exit"):
                    print("Exiting.")
                    return None
                v_str, i_str = raw.split()
                v, i = int(v_str), int(i_str)
                mv = (v, i)
            except Exception as e:
                print("Invalid input. Use: v i  (e.g., '3 4').")
                continue
            if mv not in legal:
                print("Illegal move. Legal moves:", sorted(list(legal)))
                continue
            s = Z.apply_move(s, mv)
            print_state(s)
        else:
            # bot turn
            mv = choose_move_ccp(s, theta=theta_bot, ccp=ccp, H=H, beta=beta, mode=mode_bot)
            if mv is None:
                print("Bot has no legal move. Draw.")
                return 0
            print(f"Bot plays: v={mv[0]} at i={mv[1]}")
            s = Z.apply_move(s, mv)
            print_state(s)
