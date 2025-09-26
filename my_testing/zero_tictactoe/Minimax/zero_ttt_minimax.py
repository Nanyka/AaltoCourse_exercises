
# zero_ttt_minimax.py
# Best-play (minimax) solver for Zero Tic-Tac-Toe.
# Builds an optimal policy you can reuse later (save/load as JSON.gz).

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import json, gzip
import zero_ttt_core as Z

Key = Tuple[Tuple[int,...], Tuple[int,int,int], Tuple[int,int,int], int]  # (board, invX, invO, to_move)
SYMBOL = {0:"·", 1:"X1", 2:"X2", 3:"X3", 4:"O1", 5:"O2", 6:"O3"}

def key_of(s: Z.State) -> Key:
    return (s.board, s.invX, s.invO, s.to_move)

@dataclass
class Entry:
    value: int                 # minimax value from perspective of player-to-move in this state: {-1,0,1}
    best_move: Optional[Tuple[int,int]]  # (v, i)

class MinimaxSolver:
    def __init__(self):
        self.tt: Dict[Key, Entry] = {}
        self.nodes = 0

    # -------- terminal / stalemate check --------
    def _terminal_value(self, s: Z.State) -> Optional[int]:
        term, w = Z.is_terminal(s.board)
        if term:
            # from current player's perspective
            return 1 if w == s.to_move else -1
        # Stalemate rules: no legal moves for player to move, or both inventories exhausted
        if not Z.legal_moves(s): 
            return 0
        if sum(s.invX) == 0 and sum(s.invO) == 0:
            return 0
        return None

    # -------- move ordering to speed up DFS --------
    def _ordered_moves(self, s: Z.State) -> List[Tuple[int,int]]:
        # Heuristic: immediate wins first, then blocks (opponent would win next), then others.
        moves = Z.legal_moves(s)
        if not moves: return moves
        p = s.to_move
        wins = []; blocks = []; rest = []
        # Build opponent-immediate-win detector
        def opp_wins_next(s_after: Z.State) -> bool:
            opp = s_after.to_move
            for mv in Z.legal_moves(s_after):
                b = list(s_after.board); cur = b[mv[1]]
                if cur==Z.EMPTY or (Z.OWNER[cur]!=opp and Z.VAL[cur] < mv[0]):
                    b[mv[1]] = Z.encode(opp, mv[0])
                    t, w = Z.is_terminal(tuple(b))
                    if t and w==opp: return True
            return False
        for mv in moves:
            # check win-now
            b = list(s.board); cur = b[mv[1]]
            if cur==Z.EMPTY or (Z.OWNER[cur]!=p and Z.VAL[cur] < mv[0]):
                b[mv[1]] = Z.encode(p, mv[0])
                t, w = Z.is_terminal(tuple(b))
                if t and w==p:
                    wins.append(mv); continue
            s2 = Z.apply_move(s, mv)
            if opp_wins_next(s2):
                blocks.append(mv)
            else:
                rest.append(mv)
        # Prefer higher piece values in ordering to accelerate overwrites
        def by_value(mv): return mv[0]
        return sorted(wins, key=by_value, reverse=True) + sorted(blocks, key=by_value, reverse=True) + sorted(rest, key=by_value, reverse=True)

    # -------- core DFS (negamax) --------
    def solve_state(self, s: Z.State) -> Entry:
        k = key_of(s)
        if k in self.tt:
            return self.tt[k]

        self.nodes += 1
        tv = self._terminal_value(s)
        if tv is not None:
            e = Entry(tv, None)
            self.tt[k] = e
            return e

        best_val = -2
        best_mv  = None
        for mv in self._ordered_moves(s):
            s2 = Z.apply_move(s, mv)
            child = self.solve_state(s2)
            val = -child.value
            if val > best_val:
                best_val, best_mv = val, mv
                if best_val == 1:  # cannot do better
                    break
        e = Entry(int(best_val), best_mv)
        self.tt[k] = e
        return e

    def solve_from_initial(self) -> Entry:
        return self.solve_state(Z.initial_state())

    # -------- policy interface --------
    def best_action(self, s: Z.State) -> Optional[Tuple[int,int]]:
        return self.solve_state(s).best_move

    def value(self, s: Z.State) -> int:
        return self.solve_state(s).value

    # -------- save / load --------
    def save_policy(self, path: str) -> str:
        """Save only the best action and value for each solved state as a NumPy .npy file."""
        records = []
        for k, e in self.tt.items():
            board, invX, invO, to_move = k
            rec = {
                "board": np.array(board, dtype=np.int8),
                "invX": np.array(invX, dtype=np.int8),
                "invO": np.array(invO, dtype=np.int8),
                "to_move": np.int8(to_move),
                "value": np.int8(e.value),
                "best_move": np.array(e.best_move, dtype=np.int8) if e.best_move is not None else None,
            }
            records.append(rec)

        # Save as numpy object array
        np.save(path, records, allow_pickle=True)
        return path

    @staticmethod
    def load_policy(path: str) -> "MinimaxSolver":
        """Load policy from a NumPy .npy file created with save_policy()."""
        solver = MinimaxSolver()
        data = np.load(path, allow_pickle=True)

        for rec in data:
            s = Z.State(
                tuple(rec["board"]),
                tuple(rec["invX"]),
                tuple(rec["invO"]),
                int(rec["to_move"])
            )
            k = key_of(s)
            bm = None if rec["best_move"] is None else (int(rec["best_move"][0]), int(rec["best_move"][1]))
            solver.tt[k] = Entry(int(rec["value"]), bm)

        return solver

# ---------- Convenience: functional policy wrapper ----------
class OptimalPolicy:
    """Small wrapper that consults a solved table or falls back to on-demand solve."""
    def __init__(self, solver: Optional[MinimaxSolver]=None):
        self.solver = solver or MinimaxSolver()

    def action(self, s: Z.State) -> Optional[Tuple[int,int]]:
        return self.solver.best_action(s)

    def value(self, s: Z.State) -> int:
        return self.solver.value(s)

    @classmethod
    def from_file(cls, path: str) -> "OptimalPolicy":
        return cls(MinimaxSolver.load_policy(path))

# ---------- Quick demo helpers (optional) ----------

def print_board(board: Tuple[int,...]):
    rows = []
    for r in range(3):
        i = 3*r
        rows.append(" | ".join(f"{SYMBOL[board[i+c]]:^3}" for c in range(3)))
    sep = "\n" + "-"*15 + "\n"
    print(rows[0] + sep + rows[1] + sep + rows[2])

def print_state(s: Z.State):
    print_board(s.board)
    x = f"X inv: 1×{s.invX[0]} 2×{s.invX[1]} 3×{s.invX[2]}"
    o = f"O inv: 1×{s.invO[0]} 2×{s.invO[1]} 3×{s.invO[2]}"
    print(x + "   |   " + o + f"   |   to_move: {'X' if s.to_move==Z.X else 'O'}")
