
import numpy as np
from dataclasses import dataclass
import zero_ttt_core as core
from zero_ttt_core import State, FEATURE_NAMES
import zero_ttt_core as Z

def exact_best_move(s: State):
    v, mv = core.solve_exact(s)
    return mv

def approx_best_move(s: State, theta, depth=4):
    v, mv = core.search_theta(s, depth, theta)
    return mv

def make_training_set(n=60, seed=0, steps_range=(2,8)):
    rng = np.random.default_rng(seed)
    states = []
    for k in range(n):
        steps = int(rng.integers(steps_range[0], steps_range[1]+1))
        s = core.random_reachable_state(seed=seed*997 + k*13, steps=steps)
        term,_ = core.is_terminal(s.board)
        if term: continue
        states.append(s)
    return states

# Original objective
def agreement_objective(theta, states, depth=4):
    correct=0; tot=0
    for s in states:
        mv_e = exact_best_move(s)
        if mv_e is None: 
            continue
        mv_a = approx_best_move(s, theta, depth=depth)
        correct += int(mv_a==mv_e)
        tot += 1
    return correct / max(1,tot)

# Replacee agreement_objective to: Teach θ to minimize how much value it loses vs. optimal, not just whether it matches the move.
def regret_objective(theta, states, depth=4):
    """Return NEGATIVE average regret (so NRLS maximizes this)."""
    th = np.asarray(theta, float)
    total = 0.0; n = 0
    for s in states:
        # best exact value at root (for side to move)
        q_star = -np.inf
        for mv in Z.legal_moves(s):
            v_child, _ = Z.solve_exact(Z.apply_move(s, mv))
            q_star = max(q_star, v_child)         # exact Q*(s,mv)

        # θ policy move
        _, mv_hat = Z.search_theta(s, depth=depth, theta=th)
        if mv_hat is None:  # no legal moves -> regret 0
            continue
        v_hat, _ = Z.solve_exact(Z.apply_move(s, mv_hat))

        total += (q_star - v_hat)                # regret ≥ 0
        n += 1
    if n == 0: return -0.0
    return -(total / n)

@dataclass
class NRLSResult:
    theta: np.ndarray
    value: float
    evaluations: int

def _center(lo,hi): return 0.5*(lo+hi)
def _shrink(lo,hi,v,shrink):
    span=hi-lo; half=0.5*shrink*span
    nlo, nhi = v-half, v+half
    if nhi<=nlo:
        eps=max(1e-6,1e-3*abs(v)+1e-6); nlo,nhi=v-eps,v+eps
    return float(nlo), float(nhi)

def nrls_maximize(f, bounds, levels=(5,7,9), topk=6, shrink=0.4, verbose=True):
    K=len(bounds)
    best_theta = np.array([_center(lo,hi) for lo,hi in bounds], dtype=float)
    best_val = f(best_theta); evals=1
    for L,m in enumerate(levels, start=1):
        if verbose: print(f"== Level {L}/{len(levels)} | grid={m}, topk={topk} ==")
        prefix_boxes=[tuple(bounds)]; prefix_vals=[()]
        for d in range(K):
            scored=[]; next_boxes=[]; next_prefix=[] 
            for pi,box in enumerate(prefix_boxes):
                lo,hi = box[d]; grid = np.linspace(lo,hi,m)
                for v in grid:
                    theta_proxy = np.zeros(K)
                    for j in range(K):
                        if j<d: theta_proxy[j]=prefix_vals[pi][j]
                        elif j==d: theta_proxy[j]=v
                        else: theta_proxy[j]=_center(*box[j])
                    val = f(theta_proxy); evals+=1
                    if val>best_val: best_val, best_theta = val, theta_proxy.copy()
                    scored.append((val, pi, float(v)))
            if not scored: break
            scored.sort(key=lambda t:t[0], reverse=True)
            keep = scored[:min(topk,len(scored))]
            for val,pi,v in keep:
                box=list(prefix_boxes[pi]); nlo,nhi=_shrink(*box[d], v, shrink); box[d]=(nlo,nhi)
                next_boxes.append(tuple(box))
                old=prefix_vals[pi] if len(prefix_vals)>0 else ()
                next_prefix.append(tuple(old[:d])+(v,))
            prefix_boxes, prefix_vals = next_boxes, next_prefix
        for box in prefix_boxes:
            theta_c = np.array([_center(*box[j]) for j in range(K)], dtype=float)
            val = f(theta_c); evals+=1
            if val>best_val: best_val, best_theta = val, theta_c.copy()
        bounds = [_shrink(lo,hi,best_theta[j], shrink) for j,(lo,hi) in enumerate(bounds)]
        if verbose: print(f"   best so far: {best_val:.3f} θ={best_theta}")
    return NRLSResult(theta=best_theta, value=best_val, evaluations=evals)

def train_nrls(seed=0, n_states=60, depth=4, levels=(5,7,9), topk=6, shrink=0.4):
    states = make_training_set(n=n_states, seed=seed, steps_range=(2,8))
    K = len(FEATURE_NAMES)
    bounds = [(-3,3)]*K
    f = lambda th: agreement_objective(np.asarray(th), states, depth=depth)
    res = nrls_maximize(f, bounds, levels=levels, topk=topk, shrink=shrink, verbose=True)
    return res, states

if __name__=="__main__":
    res, states = train_nrls(seed=0, n_states=40, depth=4)
    print("θ_hat:", res.theta); print("agreement:", res.value, "evals:", res.evaluations)
