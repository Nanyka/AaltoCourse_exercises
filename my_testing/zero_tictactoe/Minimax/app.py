
# app.py — FastAPI + Uvicorn site for Zero Tic‑Tac‑Toe (minimax backend)
# Run:
#   pip install fastapi uvicorn
#   uvicorn app:app --reload --host 0.0.0.0 --port 8000
from __future__ import annotations
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import sys, os

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.append(str(HERE))
if "/mnt/data" not in sys.path: sys.path.append("/mnt/data")

import zero_ttt_core as Z
from zero_ttt_minimax import MinimaxSolver, OptimalPolicy

app = FastAPI(title="Zero TTT (minimax)")
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("ZTTT_SECRET","dev-secret"))

app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

SOLVER = MinimaxSolver()
POLICY = OptimalPolicy(SOLVER)

def pack_state(s: Z.State) -> dict:
    return {
        "board": list(s.board),
        "invX": list(s.invX),
        "invO": list(s.invO),
        "to_move": "X" if s.to_move == Z.X else "O",
    }

def unpack_state(d: dict) -> Z.State:
    return Z.State(tuple(d["board"]), tuple(d["invX"]), tuple(d["invO"]), Z.X if d["to_move"]=="X" else Z.O)

def legal_moves_json(s: Z.State):
    return [{"v": v, "i": i} for (v,i) in Z.legal_moves(s)]

def winner_from_state(s: Z.State):
    term, w = Z.is_terminal(s.board)
    if term: return "X" if w==Z.X else "O"
    if (len(Z.legal_moves(s))==0) or (sum(s.invX)==0 and sum(s.invO)==0): return "draw"
    return "ongoing"

def get_game(request: Request) -> dict:
    g = request.session.get("game")
    if not g:
        s = Z.initial_state()
        g = {"human_side": "X", "state": pack_state(s)}
        request.session["game"] = g
    return g

@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")

@app.post("/api/new_game")
async def new_game(request: Request):
    data = await request.json()
    human = (data or {}).get("human_side","X").upper()
    if human not in ("X","O"): human = "X"
    s = Z.initial_state()
    g = {"human_side": human, "state": pack_state(s)}
    request.session["game"] = g
    return JSONResponse({
        "ok": True,
        "human_side": human,
        "state": g["state"],
        "legal": legal_moves_json(s),
        "result": winner_from_state(s)
    })

@app.get("/api/state")
async def api_state(request: Request):
    g = get_game(request)
    s = unpack_state(g["state"])
    return JSONResponse({
        "ok": True,
        "human_side": g["human_side"],
        "state": g["state"],
        "legal": legal_moves_json(s),
        "result": winner_from_state(s)
    })

@app.post("/api/human_move")
async def human_move(request: Request):
    g = get_game(request)
    s = unpack_state(g["state"])
    data = await request.json()
    v = int((data or {}).get("v", 0)); i = int((data or {}).get("i", -1))
    human = g["human_side"]
    if (human=="X" and s.to_move!=Z.X) or (human=="O" and s.to_move!=Z.O):
        raise HTTPException(status_code=400, detail="Not your turn.")
    if (v,i) not in Z.legal_moves(s):
        raise HTTPException(status_code=400, detail="Illegal move.")
    s2 = Z.apply_move(s, (v,i))
    g["state"] = pack_state(s2)
    request.session["game"] = g
    return JSONResponse({
        "ok": True,
        "state": g["state"],
        "legal": legal_moves_json(s2),
        "result": winner_from_state(s2)
    })

@app.post("/api/bot_move")
async def bot_move(request: Request):
    g = get_game(request)
    s = unpack_state(g["state"])
    if winner_from_state(s) != "ongoing" or not Z.legal_moves(s):
        return JSONResponse({
            "ok": True,
            "state": g["state"],
            "legal": legal_moves_json(s),
            "result": winner_from_state(s)
        })
    human = g["human_side"]
    if (human=="X" and s.to_move==Z.X) or (human=="O" and s.to_move==Z.O):
        raise HTTPException(status_code=400, detail="It's human turn.")
    mv = POLICY.action(s)
    if mv is None:
        return JSONResponse({
            "ok": True,
            "state": g["state"],
            "legal": [],
            "result": "draw",
            "bot_move": None
        })
    s2 = Z.apply_move(s, mv)
    g["state"] = pack_state(s2)
    request.session["game"] = g
    return JSONResponse({
        "ok": True,
        "state": g["state"],
        "legal": legal_moves_json(s2),
        "result": winner_from_state(s2),
        "bot_move": {"v": mv[0], "i": mv[1]}
    })
