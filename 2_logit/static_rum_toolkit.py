
# static_rum_toolkit.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

def logsumexp(v: np.ndarray) -> float:
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

def softmax_dmax(v: np.ndarray) -> np.ndarray:
    m = np.max(v)
    z = np.exp(v - m)
    return z / np.sum(z)

class model:
    def __init__(self, label='noname', nalt=1, nattr=1, attr=None, param=None, st=None, sigma=1.0, spec='linear'):
        if st is None: st = [1,2]
        if attr is None: attr = np.ones((nalt, nattr))
        if param is None: param = np.ones(nattr)
        self.label, self.nalt, self.nattr = label, int(nalt), int(nattr)
        self.attr, self.param = np.asarray(attr, float), np.asarray(param, float)
        self.st, self.sigma, self.spec = list(st), float(sigma), spec
        self._chpr, self._sim = {}, {}
        self._validate()

    def _validate(self):
        assert self.attr.shape == (self.nalt, self.nattr)
        assert self.param.shape == (self.nattr,)
        assert self.sigma > 0
        for x in self.st: assert x in (1,2)
        if self.spec == 'nonlinear' and np.any(self.attr <= 0):
            raise ValueError('Nonlinear spec requires positive attributes.')

    def __setattr__(self, n, v):
        super().__setattr__(n, v)
        if n in ('attr','param','nalt','nattr','sigma','spec','st'):
            if all(hasattr(self,k) for k in ('nalt','nattr','attr','param')):
                try: self._validate()
                except Exception: pass

    def __str__(self):
        return f"Model: {self.label}\nspec={self.spec}, sigma={self.sigma}\nnalt={self.nalt}, nattr={self.nattr}, states={self.st}\nattr (Y):\n{self.attr}\nparam (β): {self.param}"

    def beta_coef(self, x:int):
        return self.param if x==1 else (self.param/2.0 if x==2 else self.param)

    def utility(self, x:int):
        b = self.beta_coef(x)
        return self.attr @ b if self.spec=='linear' else (np.log(self.attr) @ b)

    def save_chpr(self, x:int, p:np.ndarray):
        self._chpr[int(x)] = np.asarray(p, float)

    @property
    def choice_prob(self): return self._chpr

    @property
    def sim_data(self): return self._sim

def model_solve(m:model):
    res = {}
    for x in m.st:
        u = m.utility(x)
        p = softmax_dmax(u / m.sigma)
        m.save_chpr(x, p)
        res[x] = p
    return res

def model_simulate(m:model, N_per_state=2000, seed=123):
    if not m.choice_prob: model_solve(m)
    rng = np.random.default_rng(seed)
    out = {}
    for x in m.st:
        out[x] = rng.choice(m.nalt, size=N_per_state, p=m.choice_prob[x])
    m._sim = out
    return out

def plot_choice_probabilities(m:model):
    if not m.choice_prob: model_solve(m)
    for x in m.st:
        plt.figure()
        plt.title(f'Choice probabilities — state {x} (spec={m.spec}, σ={m.sigma})')
        plt.bar(np.arange(m.nalt), m.choice_prob[x])
        plt.xlabel('Alternative'); plt.ylabel('P(d|x)')
        plt.xticks(np.arange(m.nalt), [f'alt{j+1}' for j in range(m.nalt)])
        plt.show()

def plot_simulated_data(m:model):
    if not m.sim_data: raise RuntimeError('No simulated data; call model_simulate(...) first.')
    for x in m.st:
        plt.figure()
        plt.title(f'Histogram of simulated choices — state {x}')
        plt.hist(m.sim_data[x], bins=np.arange(-0.5, m.nalt+0.5, 1))
        plt.xlabel('Alternative (index)'); plt.ylabel('Count')
        plt.xticks(np.arange(m.nalt), [f'alt{j+1}' for j in range(m.nalt)])
        plt.show()

# def plot_dashboard(m:model, N_per_state=1500, seed=42):
#     import pandas as pd
#     from caas_jupyter_tools import display_dataframe_to_user
#     attr_df = pd.DataFrame(m.attr, columns=[f'a{j+1}' for j in range(m.nattr)]); attr_df.index=[f'alt{j+1}' for j in range(m.nalt)]
#     display_dataframe_to_user('Alternative attributes (Y)', attr_df)
#     beta_df = pd.DataFrame([m.beta_coef(x) for x in m.st], index=[f'state {x}' for x in m.st], columns=[f'b{j+1}' for j in range(m.nattr)])
#     display_dataframe_to_user('Structural coefficients by state (β_x)', beta_df)
#     model_solve(m); plot_choice_probabilities(m)
#     model_simulate(m, N_per_state=N_per_state, seed=seed); plot_simulated_data(m)
