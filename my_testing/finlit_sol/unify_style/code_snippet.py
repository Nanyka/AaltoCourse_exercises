
import numpy as np
from scipy.special import logsumexp

# Load saved objects
CCP = np.load("CCP_hat.npy")
V = np.load("V_hat.npy")

# (For a full re-solve, re-run the notebook script that built flow_utility() and solve_v().)
print("Loaded CCP shape:", CCP.shape, "V shape:", V.shape)
