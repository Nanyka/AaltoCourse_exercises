
import numpy as np
from dataclasses import dataclass
from numpy.polynomial.hermite import hermgauss

def gh_nodes_weights(n=10):
    nodes, weights = hermgauss(n)   # e^{-x^2}
    weights = weights / np.sqrt(np.pi)  # standard normal
    return nodes, weights

@dataclass
class Spec:
    beta: float       # discount
    sigma: float      # CRRA over consumption
    R_f: float        # safe gross return
    mu_lnR: float     # mean of log risky gross return
    sigma_lnR: float  # std of log risky gross return
    # State-dependent fee parameters (functions of a via log(a))
    gamma0: float     # k_fix(a) = exp(gamma0 + gamma1 * ln a)
    gamma1: float
    delta0: float     # tau(a) = tau_min + tau_range * sigmoid(delta0 + delta1 * ln a)
    delta1: float
    tau_min: float = 1e-4
    tau_range: float = 0.10

class NFXPFinLitConsBudgetStateFx:
    """
    DP with consumption utility and **state-dependent** budget-embedded costs.
      k_fix(a) = exp(gamma0 + gamma1 * ln a)
      tau(a)   = tau_min + tau_range * sigmoid(delta0 + delta1 * ln a)

    States: a (assets grid)
    Actions: x in x_grid (risky share), s in s_grid (savings rate)
    Budget:
        res_gross = a * r_p(x) + y,  r_p(x) = x*R + (1-x)*R_f
        if x>0:
            res_net  = (1 - tau(a)) * res_gross - k_fix(a)
        else:
            res_net  = res_gross
        c     = (1 - s) * res_net
        a'    = s * res_net
    u(c) = CRRA(c; sigma). EV1 shocks => logit CCPs over joint (x, s).
    """
    def __init__(self, a_grid: np.ndarray, y_bar: float, x_grid: np.ndarray, s_grid: np.ndarray,
                 gh_order: int = 8):
        self.a = a_grid
        self.na = len(a_grid)
        self.y = float(y_bar)
        self.xgrid = x_grid
        self.ndx = len(x_grid)
        self.sgrid = s_grid
        self.nds = len(s_grid)
        self.nodes, self.weights = gh_nodes_weights(gh_order)

    @staticmethod
    def crra(c, sigma):
        eps = 1e-10
        c = np.maximum(c, eps)
        if abs(sigma - 1.0) < 1e-8:
            return np.log(c)
        return (np.power(c, 1.0 - sigma) - 1.0) / (1.0 - sigma)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fee_functions(self, a_vec, spec: Spec):
        # a_vec shape: (na,)
        ln_a = np.log(np.maximum(a_vec, 1e-8))
        k_fix_a = np.exp(spec.gamma0 + spec.gamma1 * ln_a)   # (na,)
        tau_a = spec.tau_min + spec.tau_range * self.sigmoid(spec.delta0 + spec.delta1 * ln_a)
        return k_fix_a, tau_a

    def value_iteration(self, spec: Spec, tol=1e-6, maxit=400):
        R_vals = np.exp(spec.mu_lnR + spec.sigma_lnR * self.nodes)  # (nR,)
        V = np.zeros(self.na)
        k_fix_a, tau_a = self.fee_functions(self.a, spec)            # (na,), (na,)

        for it in range(maxit):
            def V_interp(a_prime):
                a_min, a_max = self.a[0], self.a[-1]
                a_prime = np.clip(a_prime, a_min, a_max)
                return np.interp(a_prime, self.a, V)

            Q = np.zeros((self.na, self.ndx, self.nds))

            for j, x in enumerate(self.xgrid):
                rp = x * R_vals + (1 - x) * spec.R_f               # (nR,)
                res_gross = np.outer(self.a, rp) + self.y           # (na, nR)
                part = (x > 0).astype(float)

                # Broadcast k_fix(a) and tau(a) across return nodes
                k_fix_mat = k_fix_a[:, None] * part
                tau_mat   = tau_a[:, None]   * part

                res_net = res_gross * (1.0 - tau_mat) - k_fix_mat
                res_net = np.maximum(res_net, 1e-10)

                for k, s in enumerate(self.sgrid):
                    c_next = (1.0 - s) * res_net
                    a_next = s * res_net

                    U = self.crra(c_next, spec.sigma)
                    Vcont = np.array([V_interp(w) for w in a_next.ravel()]).reshape(a_next.shape)
                    EU = (U + spec.beta * Vcont) @ self.weights       # (na,)
                    Q[:, j, k] = EU

            # log-sum-exp over joint actions
            Q2 = Q.reshape(self.na, -1)
            row_max = Q2.max(axis=1, keepdims=True)
            V_new = (row_max + np.log(np.exp(Q2 - row_max).sum(axis=1, keepdims=True))).ravel()

            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        Q2 = Q.reshape(self.na, -1)
        row_max = Q2.max(axis=1, keepdims=True)
        denom = np.exp(Q2 - row_max).sum(axis=1, keepdims=True)
        CCP_flat = np.exp(Q2 - row_max) / denom
        CCP = CCP_flat.reshape(self.na, self.ndx, self.nds)
        pol_flat = CCP_flat.argmax(axis=1)
        pol_x_idx = (pol_flat // self.nds)
        pol_s_idx = (pol_flat % self.nds)
        pol_x = self.xgrid[pol_x_idx]
        pol_s = self.sgrid[pol_s_idx]
        return V, Q, CCP, pol_x, pol_s

    def ccp_x_marginal(self, a_obs: np.ndarray, CCP: np.ndarray):
        idx = np.searchsorted(self.a, a_obs, side='left')
        idx = np.clip(idx, 0, self.na - 1)
        return CCP[idx, :, :].sum(axis=2)
