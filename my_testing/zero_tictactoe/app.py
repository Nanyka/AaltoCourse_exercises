
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys, os

# allow import of game core
sys.path.append("/mnt/data")
import zero_ttt_core as Z

class Move(BaseModel):
    v:int; i:int

class StateModel(BaseModel):
    board: List[int]; invX: List[int]; invO: List[int]; to_move: int

class NewGameReq(BaseModel):
    human: str = "X"; theta: Optional[List[float]] = None; depth: int = 4

class HumanMoveReq(BaseModel):
    state: StateModel; move: Move

class BotMoveReq(BaseModel):
    state: StateModel; theta: Optional[List[float]] = None; depth: int = 4

def state_to_model(s: Z.State) -> StateModel:
    return StateModel(board=list(s.board), invX=list(s.invX), invO=list(s.invO), to_move=s.to_move)

def model_to_state(m: StateModel) -> Z.State:
    return Z.State(tuple(m.board), tuple(m.invX), tuple(m.invO), m.to_move)

DEFAULT_THETA = [2.2,0.8,0.2, 1.2,-1.5, 1.4,1.1,0.6, 0,0,0, 0.4]

app = FastAPI(title="Zero TTT — NRLS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/health")
def health(): return {"ok": True}

@app.post("/api/new")
def new_game(req: NewGameReq):
    s = Z.initial_state()
    return {"state": state_to_model(s), "human": req.human.upper(), "theta": req.theta or DEFAULT_THETA, "depth": req.depth, "terminal": False, "winner": 0}

@app.post("/api/human-move")
def human_move(req: HumanMoveReq):
    s = model_to_state(req.state); mv = (req.move.v, req.move.i)
    if mv not in Z.legal_moves(s): return {"error":"illegal move"}
    s2 = Z.apply_move(s, mv)
    term, w = Z.is_terminal(s2.board)
    terminal = bool(term) or (sum(s2.invX)==0 and sum(s2.invO)==0)
    return {"state": state_to_model(s2), "terminal": terminal, "winner": int(w if term else 0)}

@app.post("/api/bot-move")
def bot_move(req: BotMoveReq):
    s = model_to_state(req.state)
    theta = np.array(req.theta if req.theta is not None else DEFAULT_THETA, dtype=float)
    val, mv = Z.search_theta(s, depth=req.depth, theta=theta)
    if mv is None:
        return {"state": state_to_model(s), "move": None, "terminal": True, "winner": 0}
    s2 = Z.apply_move(s, mv)
    term, w = Z.is_terminal(s2.board)
    terminal = bool(term) or (sum(s2.invX)==0 and sum(s2.invO)==0)
    return {"state": state_to_model(s2), "move": {"v": mv[0], "i": mv[1]}, "eval": float(val), "terminal": terminal, "winner": int(w if term else 0)}
