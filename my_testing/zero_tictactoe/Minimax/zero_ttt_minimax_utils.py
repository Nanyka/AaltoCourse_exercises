
# zero_ttt_minimax_utils.py
# Helpers to test the minimax model and let a human play against it.
# Works with: zero_ttt_core.py, zero_ttt_minimax.py

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

import zero_ttt_core as Z
from zero_ttt_minimax import MinimaxSolver, OptimalPolicy

# ---------------- Pretty print ----------------
SYMBOL = {0:"·", 1:"X1", 2:"X2", 3:"X3", 4:"O1", 5:"O2", 6:"O3"}

def board_to_str(board: Tuple[int,...]) -> str:
    rows = []
    for r in range(3):
        i = 3*r
        rows.append(" | ".join(f"{SYMBOL[board[i+c]]:^3}" for c in range(3)))
    sep = "\n" + "-"*15 + "\n"
    return rows[0] + sep + rows[1] + sep + rows[2]

def print_state(s: Z.State):
    print(board_to_str(s.board))
    x = f"X inv: 1×{s.invX[0]} 2×{s.invX[1]} 3×{s.invX[2]}"
    o = f"O inv: 1×{s.invO[0]} 2×{s.invO[1]} 3×{s.invO[2]}"
    print(x + "   |   " + o + f"   |   to_move: {'X' if s.to_move==Z.X else 'O'}")

def result_text(winner: int) -> str:
    return "X wins" if winner==Z.X else "O wins" if winner==Z.O else "Draw"

# ---------------- Core helpers ----------------
def winner_from_state(s: Z.State) -> int:
    term, w = Z.is_terminal(s.board)
    if term:
        return w
    if (len(Z.legal_moves(s))==0) or (sum(s.invX)==0 and sum(s.invO)==0):
        return 0
    return -1  # ongoing

def random_move(s: Z.State, rng: Optional[np.random.Generator]=None) -> Optional[Tuple[int,int]]:
    rng = rng or np.random.default_rng()
    legal = Z.legal_moves(s)
    if not legal: return None
    return legal[int(rng.integers(0, len(legal)))]

def play_game(policy_X: OptimalPolicy, policy_O: OptimalPolicy, verbose: bool=False):
    """Play a full game policy_X vs policy_O. Return (winner, history, final_state)."""
    s = Z.initial_state()
    history = []
    while True:
        w = winner_from_state(s)
        if w != -1:
            return w, history, s
        pol = policy_X if s.to_move==Z.X else policy_O
        mv = pol.action(s)
        if mv is None:  # stalemate
            return 0, history, s
        history.append((s, mv))
        s = Z.apply_move(s, mv)
        if verbose:
            print_state(s)

# ---------------- Human vs Model (CLI) ----------------
def human_vs_model_cli(human: str="X", solver: OptimalPolicy=None):
    """Simple terminal interaction: human vs optimal policy. Use in a terminal (not great inside Jupyter)."""
    human = human.upper()
    assert human in ("X","O")
    solver = solver or OptimalPolicy()
    # pi = OptimalPolicy(solver)
    s = Z.initial_state()
    print("Welcome to Zero TTT (minimax). You are", human)
    print("Cells indexed 0..8 as: 0 1 2 / 3 4 5 / 6 7 8\n")
    print_state(s)
    while True:
        w = winner_from_state(s)
        if w != -1:
            print("Game over:", result_text(w))
            return w
        if (human=="X" and s.to_move==Z.X) or (human=="O" and s.to_move==Z.O):
            legal = set(Z.legal_moves(s))
            try:
                raw = input("Your move (v i), or 'q' to quit: ").strip()
                if raw.lower() in ("q","quit","exit"):
                    print("Quit.")
                    return None
                v_str, i_str = raw.split()
                v, i = int(v_str), int(i_str)
                mv = (v,i)
            except Exception:
                print("Invalid input. Use e.g. '3 4' to place value 3 at index 4.")
                continue
            if mv not in legal:
                print("Illegal. Legal moves:", sorted(list(legal)))
                continue
            s = Z.apply_move(s, mv)
            print_state(s)
        else:
            mv = solver.action(s)
            assert mv is not None
            print(f"Model plays: v={mv[0]} at i={mv[1]}")
            s = Z.apply_move(s, mv)
            print_state(s)

# ---------------- Human vs Model (Jupyter-friendly session) ----------------
@dataclass
class Session:
    solver: MinimaxSolver
    policy: OptimalPolicy
    human_side: int
    state: Z.State

def start_session(human: str="X", solver: Optional[MinimaxSolver]=None) -> Session:
    human = human.upper()
    assert human in ("X","O")
    solver = solver or MinimaxSolver()
    return Session(solver, OptimalPolicy(solver), Z.X if human=="X" else Z.O, Z.initial_state())

def show(sess: Session):
    print_state(sess.state)

def legal(sess: Session) -> List[Tuple[int,int]]:
    return Z.legal_moves(sess.state)

def human_move(sess: Session, v: int, i: int) -> str:
    if sess.state.to_move != sess.human_side:
        return "Not your turn."
    mv = (v,i)
    L = set(Z.legal_moves(sess.state))
    if mv not in L:
        return f"Illegal move. Legal: {sorted(list(L))}"
    sess.state = Z.apply_move(sess.state, mv)
    w = winner_from_state(sess.state)
    return f"OK. {result_text(w) if w!=-1 else 'Game continues.'}"

def model_move(sess: Session) -> str:
    if sess.state.to_move == sess.human_side:
        return "It's your turn."
    mv = sess.policy.action(sess.state)
    if mv is None:
        return "Model has no legal move. Draw."
    sess.state = Z.apply_move(sess.state, mv)
    w = winner_from_state(sess.state)
    return f"Model played v={mv[0]} i={mv[1]}. {result_text(w) if w!=-1 else 'Game continues.'}"
