
import numpy as np
import zero_ttt_core as Z
from zero_ttt_core import (
    State, initial_state, legal_moves, apply_move, is_terminal,
    search_theta, solve_exact, OWNER, VAL, EMPTY, X, O
)

def board_to_str(board):
    def tok(c):
        if c == EMPTY: return '.'
        return f"{'X' if OWNER[c]==X else 'O'}{VAL[c]}"
    s = [tok(c) for c in board]
    s = [t.rjust(2) for t in s]
    rows = [' '.join(s[0:3]), ' '.join(s[3:6]), ' '.join(s[6:9])]
    return '\n'.join(rows)

def print_board(board):
    print(board_to_str(board))

def index_map_str():
    return " 0  1  2\n 3  4  5\n 6  7  8"

def print_trajectory(traj):
    for t, (st, mv) in enumerate(traj, start=1):
        who = 'X' if st.to_move==X else 'O'
        print(f"Move {t}: {who} plays (value={mv[0]}, index={mv[1]})")
        print(board_to_str(st.board)); print()

def choose_move_theta(state, theta, depth=4):
    val, mv = search_theta(state, depth=depth, theta=np.asarray(theta, dtype=float))
    return mv

def choose_move_exact(state):
    val, mv = solve_exact(state)
    return mv

def choose_move_random(state, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    mv_list = legal_moves(state)
    return None if not mv_list else tuple(mv_list[int(rng.integers(0, len(mv_list)))])

def play_game(theta_X=None, theta_O=None, depth_X=4, depth_O=4,
              use_exact_X=False, use_exact_O=False, seed=None,
              verbose=True):
    rng = np.random.default_rng(seed)
    s = initial_state()
    traj = []
    if verbose:
        print("Index map:\n" + index_map_str() + "\n")
        print("Start:\n" + board_to_str(s.board) + "\n")
    while True:
        term, w = is_terminal(s.board)
        if term:
            if verbose:
                print(("X wins!" if w==X else "O wins!") + "\nFinal:\n" + board_to_str(s.board))
            return w, traj, s
        if sum(s.invX)==0 and sum(s.invO)==0:
            if verbose:
                print("Draw (inventories empty).\nFinal:\n" + board_to_str(s.board))
            return 0, traj, s

        if s.to_move == X:
            if use_exact_X:
                mv = choose_move_exact(s)
            elif theta_X is not None:
                mv = choose_move_theta(s, theta_X, depth=depth_X)
            else:
                mv = choose_move_random(s, rng)
        else:
            if use_exact_O:
                mv = choose_move_exact(s)
            elif theta_O is not None:
                mv = choose_move_theta(s, theta_O, depth=depth_O)
            else:
                mv = choose_move_random(s, rng)

        if mv is None:
            if verbose: print("No legal moves. Draw.")
            return 0, traj, s

        if verbose:
            who = 'X' if s.to_move==X else 'O'
            print(f"{who} plays (value={mv[0]}, index={mv[1]}):")
        traj.append((s, mv))
        s = apply_move(s, mv)
        if verbose:
            print(board_to_str(s.board)); print()

def human_vs_bot(theta, bot_player='O', depth=4, seed=None):
    bot = X if str(bot_player).upper()=='X' else O
    human = O if bot==X else X

    s = initial_state()
    print("Index map:\n" + index_map_str() + "\n")
    print("Start:\n" + board_to_str(s.board) + "\n")

    def parse_input(s, lm):
        while True:
            raw = input(f"Your move ({'X' if human==X else 'O'}). Enter index and value (e.g., '4 3') | legal={lm}: ").strip()
            parts = raw.replace(',', ' ').split()
            if len(parts) != 2:
                print("Please enter exactly two numbers: index(0..8) and value(1,2,3).")
                continue
            a, b = parts
            try:
                a = int(a); b = int(b)
            except ValueError:
                print("Please enter integers.")
                continue
            candidates = []
            if (b in (1,2,3)) and (0 <= a <= 8): candidates.append((b,a))
            if (a in (1,2,3)) and (0 <= b <= 8): candidates.append((a,b))
            for mv in candidates:
                if mv in lm: return mv
            print("That pair isn't legal. Try again.")

    while True:
        term, w = is_terminal(s.board)
        if term:
            print(("Bot wins!" if w==bot else "You win!") + "\nFinal:\n" + board_to_str(s.board))
            return w
        if sum(s.invX)==0 and sum(s.invO)==0:
            print("Draw (inventories empty).")
            print("Final:\n" + board_to_str(s.board))
            return 0

        if s.to_move == human:
            lm = legal_moves(s)
            if not lm:
                print("No legal moves. Draw."); return 0
            mv = parse_input(s, lm)
            s = apply_move(s, mv)
            print("You played:", mv)
            print(board_to_str(s.board)); print()
        else:
            mv = choose_move_theta(s, theta, depth=depth)
            if mv is None:
                print("Bot has no legal moves. Draw."); return 0
            s = apply_move(s, mv)
            print(f"Bot plays {mv}:\n{board_to_str(s.board)}\n")
