
import json, gzip, numpy as np
from typing import List, Dict, Tuple, Optional
import zero_ttt_core as Z
from ccp_zero_ttt import Datum, action_id, id_to_action

# ---------- Dataset <-> arrays / json ----------
def dataset_to_arrays(data: List[Datum]):
    N = len(data)
    boards = np.zeros((N, 9), dtype=np.int8)
    invX   = np.zeros((N, 3), dtype=np.int8)
    invO   = np.zeros((N, 3), dtype=np.int8)
    to_mv  = np.zeros(N, dtype=np.int8)
    a_id   = np.zeros(N, dtype=np.int16)
    for k,d in enumerate(data):
        boards[k] = np.array(d.state.board, dtype=np.int8)
        invX[k]   = np.array(d.state.invX, dtype=np.int8)
        invO[k]   = np.array(d.state.invO, dtype=np.int8)
        to_mv[k]  = d.state.to_move
        a_id[k]   = d.a_id
    return boards, invX, invO, to_mv, a_id

def arrays_to_dataset(boards, invX, invO, to_mv, a_id) -> List[Datum]:
    out = []
    for k in range(len(a_id)):
        s = Z.State(tuple(int(x) for x in boards[k]),
                    tuple(int(x) for x in invX[k]),
                    tuple(int(x) for x in invO[k]),
                    int(to_mv[k]))
        out.append(Datum(s, int(a_id[k])))
    return out

def save_dataset_npz(path: str, data: List[Datum]):
    boards, invX, invO, to_mv, a_id = dataset_to_arrays(data)
    np.savez_compressed(path, boards=boards, invX=invX, invO=invO, to_move=to_mv, a_id=a_id)
    return path

def load_dataset_npz(path: str) -> List[Datum]:
    Znp = np.load(path)
    return arrays_to_dataset(Znp['boards'], Znp['invX'], Znp['invO'], Znp['to_move'], Znp['a_id'])

def save_dataset_jsonl(path: str, data: List[Datum], compress: bool=True):
    opener = gzip.open if compress and (path.endswith('.gz') or path.endswith('.gzip')) else open
    mode = 'wt'
    with opener(path, mode, encoding='utf-8') as f:
        for d in data:
            rec = {
                "board": list(d.state.board),
                "invX":  list(d.state.invX),
                "invO":  list(d.state.invO),
                "to_move": int(d.state.to_move),
                "a_id": int(d.a_id)
            }
            f.write(json.dumps(rec) + "\n")
    return path

def load_dataset_jsonl(path: str) -> List[Datum]:
    opener = gzip.open if (path.endswith('.gz') or path.endswith('.gzip')) else open
    out = []
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            s = Z.State(tuple(rec["board"]), tuple(rec["invX"]), tuple(rec["invO"]), int(rec["to_move"]))
            out.append(Datum(s, int(rec["a_id"])))
    return out

# ---------- CCP save / load ----------
def save_ccp_jsonl(path: str, ccp: Dict, compress: bool=True):
    opener = gzip.open if compress and (path.endswith('.gz') or path.endswith('.gzip')) else open
    mode = 'wt'
    with opener(path, mode, encoding='utf-8') as f:
        for key, probs in ccp.items():
            player, (board, invX, invO) = key
            rec = {"p": int(player), "b": list(board), "x": list(invX), "o": list(invO), "probs": [float(x) for x in probs]}
            f.write(json.dumps(rec) + "\n")
    return path

def load_ccp_jsonl(path: str) -> Dict:
    opener = gzip.open if (path.endswith('.gz') or path.endswith('.gzip')) else open
    out = {}
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            key = (int(rec["p"]), (tuple(rec["b"]), tuple(rec["x"]), tuple(rec["o"])))
            out[key] = np.array(rec["probs"], dtype=float)
    return out

# ---------- Theta save / load ----------
def save_theta(path: str, theta):
    theta = np.asarray(theta, dtype=float)
    np.save(path, theta)
    return path

def load_theta(path: str):
    return np.load(path)
