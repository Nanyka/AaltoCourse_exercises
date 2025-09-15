#%%
import numpy as np
import copy
import matplotlib.pyplot as plt
from dataclasses import dataclass
#%%
@dataclass(init=True)
class sgame():
  '''Simple static entry game model class, see Su2014'''
  # default model parameters
  alpha: float = 5      # parameter for monopoly profits
  beta: float = -11     # parameter for duopoly profits
  x_a: float = 0.52     # size of firm a
  x_b: float = 0.22     # size of firm b

  # parameters for algorithms
  maxiter: int = 100    # max nr of iterations
  tol: float = 1e-10    # tolerance level
  verbose: int = 1      # verbosity level

  def __str__(self):
    '''String representation of the sgame model object'''
    # id() is unique identifier for the variable (reference), convert to hex
    s = f'''
    Model from sgame class with id={hex(id(self))} with attributes:
     alpha = {self.alpha: 5.2f} (Monopoly profits)
      beta = {self.beta: 5.2f} (Duopoly profits)
       x_a = {self.x_a: 5.2f} (type, firm_a)
       x_b = {self.x_b: 5.2f} (type, firm b)
      '''
    return s

  def __repr__(self):
    '''Print for sgame model object'''
    return self.__str__()

  def br_a(self, p_b):
    '''best response function, firm a'''
    p_a = 1 / (1 + np.exp(-self.x_a * self.alpha + p_b * self.x_a * (self.alpha - self.beta)))
    return p_a

  def br_b(self, p_a):
    '''best response function, firm b'''
    p_b = 1.0 / (1 + np.exp(-self.x_b * self.alpha + p_a * self.x_b * (self.alpha - self.beta)))
    return p_b

  def br2_a(self, p_a):
    ''' second order best response function, firm a '''
    p_b = self.br_b(p_a)
    p_a = self.br_a(p_b)
    return p_a

  def solve(self, method='fxp'):
    ''' Solve the model for all equilibria 
    Inputs: model and method = 'fxp' or 'lin'
    '''
    if method == 'fxp':
      # computing second order best response fixed points
      return self.fxp_eqb(self.br2_a)
    elif method == 'lin':
      # piecewise linear approximation approach (not yet implemented)
      pass
    else:
      raise ValueError(f"Unknown method: {method}, must be 'fxp' for second order best response approach or 'lin' for piecewise linear approximation approach")

  def fxp_eqb(self, fn):
    ''' Procedure to find all equilibria of the static game model using fixed point approach
    '''

    # internal functions (all fxp_eqb defined inside internal functions, careful)
    def find_stable_eqb(p00):
      '''Successive approximations'''
      p0 = p00
      for iter in range(self.maxiter):
        p = fn(p0)
        err = abs(p - p0)
        if err < self.tol:
          if self.verbose > 1:
            print(f'Stable equilibrium starting from {p00:1.3f} found after {iter} iterations, p_a={p:1.3f} err = {err:1.4e}')
          break
        p0 = p
      else:
        print(f'find_stable_eqb did not converge in {self.maxiter} iterations, last err = {err:1.4e}')
      return p

    def bisections(a0, b0):
      '''Bisections for find unstable fixed point'''
      a, b = a0, b0
      fun = lambda x: fn(x) - x
      for iter in range(self.maxiter):
        err = abs(b - a)
        if err < self.tol:
          if self.verbose > 1:
            print(f'Equilibrium found on [{a0:1.3f},{b0:1.3f}] after {iter} iterations, p_a={x:1.3f} err = {err:1.4e}')
          break
        x = (a + b) / 2
        a, b = (x, b) if (fun(a) * fun(x) > 0) else (a, x)
      else:
        print(f'bisections did not converge in {self.maxiter} iterations, last err = {err:1.4e}')
      return x

    # first find stable equilibrium approach from above and below
    pmin = find_stable_eqb(0.)
    pmax = find_stable_eqb(1.)
    if abs(pmin - pmax) > 2 * self.tol:
      # more than one stable equilibrium found.
      step = 10 * self.tol
      pmid = bisections(pmin + step, pmax - step)
      return np.array([pmin, pmid, pmax])
    else:
      # unique equilibrium
      return np.array([pmin])

  def simulate(self, eqb=0, N=10000):
    ''' Procedure to simulate data from the static game model
    Inputs: model, eqb (equilibrium selection rule, 0=lowest p_a
    1=middle p_a, 2=highest p_a), N (number of observations)
    Outputs: data (Nx2 array of simulated entry decisions for firm a and b)
    '''
    randnum = np.random.rand(N, 2)
    # Solve for equilibrium probabilities
    pa_all = self.solve(method='fxp')
    pb_all = self.br_b(pa_all)
    neqb = np.size(pa_all)
    eqb = min(eqb, neqb - 1)
    p_a = pa_all[eqb]
    p_b = self.br_b(p_a)

    print('pa:', p_a)
    print('pb:', p_b)

    dta = np.ones([N, 2])
    dta[:, 0] = 1 * (randnum[:, 0] < p_a)
    dta[:, 1] = 1 * (randnum[:, 1] < p_b)
    return dta

  def logl(self, data, theta):
    ''' Log likelihood function for NFXP estimation static entry game
    theta = (a,b)
    '''
    m = copy.deepcopy(self)
    m.a = theta[0]
    m.b = theta[1]
    d_a = data[:, 0]
    d_b = data[:, 1]
    pa = m.solve(method='fxp')
    pb = m.br_b(pa)
    neqb = np.size(pa)

    logl_ieqb = np.ones([neqb, 1])
    # compute log likelihood associated with each equilibrium
    for ieqb in range(neqb):
      logl_i= d_a*np.log(pa[ieqb]) + (1-d_a)*np.log(1-pa[ieqb]) \
          + d_b*np.log(pb[ieqb]) + (1-d_b)*np.log(1-pb[ieqb])
      logl_ieqb[ieqb, :] = sum(logl_i)
    logl = max(logl_ieqb)
    return logl

# %%
def plot_br(model: sgame, what='br'):
  ''' Plot best response functions 
  Inputs: model and what to plot (br or br2)
  '''
  assert what in ['br', 'br2'], "Argument 'what' must be 'br' or 'br2'"
  # Part 1: data
  pvec = np.arange(0, 1, 0.001, dtype=float) # dense grid
  pa = model.solve(method='fxp') # equilibria

  if what == 'br':
    l1x, l1y = model.br_a(pvec), pvec
    l2x, l2y, sty2 = pvec, model.br_b(pvec), '-b'
    sty1, lab1 = '-r', r'$\psi_a(p_b)$'
    sty2, lab2 = '-b', r'$\psi_b(p_a)$'
    pb = model.br_b(pa) # for equilibria
  elif what == 'br2':
    l1x, l1y = pvec, model.br2_a(pvec)
    l2x, l2y = pvec, pvec
    sty1, lab1 = '-r', r'${\psi_a(\psi_b(p_a))}$'
    sty2, lab2 = ':k', r'45$^\circ$'
    pb = pa # for equilibria
  else:
    raise ValueError("Argument 'what' must be 'br' or 'br2'")

  # Part 2: plot
  _, ax = plt.subplots(1, 1)
  ax.plot(l1x, l1y, sty1, label=lab1)
  ax.plot(l2x, l2y, sty2, label=lab2)
  ax.plot(pa, pb, 'ok')
  # decorate
  ax.set_xlim(0.0, 1)
  ax.set_ylim(0.0, 1)
  ax.set_aspect('equal', adjustable='box')
  if what == 'br':
    ax.set_title('Best response functions, firm a and firm b')
    ax.set_xlabel('$p_a$')
    ax.set_ylabel('$p_b$')
    for xy in zip(pa, pb):
      ax.annotate('    (%5.3f, %5.3f)' % xy, xy=xy, textcoords='data')
  elif what == 'br2':
    ax.set_title('Second order best response function, firm a')
    ax.set_xlabel('$p_a$')
    ax.set_ylabel('second order best response')
    for xy in zip(pa, pb):
      pb = model.br_b(xy[1])
      ax.annotate('    (%5.3f, %5.3f)' % (xy[0], pb), xy=xy, textcoords='data')
  leg = ax.legend()
  leg.set_frame_on(False)
  # save figure as file
  plt.savefig(fname=f'{what}.png', dpi=150)
